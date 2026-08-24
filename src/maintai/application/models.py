"""Model registry application service (P0 candidate → demo deploy).

Coordinates the P0 registry loop: register a successful experiment's recommended
``ModelRun`` as a new ``candidate`` version in MLflow and mirror it in the
business DB; then deploy exactly one version of a name as the ``demo_deployed``
serving model.

P0 never promotes to ``champion`` or ``Production``: those transitions require
human approval and belong to P1. Every returned value is JSON-safe, exposes only
``models:/`` MLflow URIs (never filesystem paths), and stores only the controlled
package-artifact basename internally (never surfaced in ``get``/``list``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    DEPLOYMENT_STATUS_CANDIDATE,
    DEPLOYMENT_STATUS_DEMO_DEPLOYED,
    EXPERIMENT_STATUS_SUCCEEDED,
    RegisteredModel,
    new_id,
)
from maintai.ml import package
from maintai.ml.package import ArtifactError
from maintai.mlops.registry import CANDIDATE_ALIAS, MLflowRegistry, RegistryError

_MODELS_PREFIX = "models:/"


class ModelRegistryServiceError(Exception):
    """Base class for model-registry application-service errors."""


class ModelRunNotFoundError(ModelRegistryServiceError):
    """Raised when a model-run id does not exist."""


class ExperimentNotEligibleError(ModelRegistryServiceError):
    """Raised when a model run is not the recommended run of a successful experiment."""


class RegisterConflictError(ModelRegistryServiceError):
    """Raised when a model run was already registered under a different name."""


class RegisteredModelNotFoundError(ModelRegistryServiceError):
    """Raised when a registered-model id does not exist."""


class ModelRegistryService:
    """Application service for P0 candidate registration and demo deployment."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        model_repository: ModelRepository,
        experiment_repository: ExperimentRepository,
        audit: AuditService,
        registry: MLflowRegistry,
        artifact_root: str | Path,
    ) -> None:
        self._session_factory = session_factory
        self._model_repository = model_repository
        self._experiment_repository = experiment_repository
        self._audit = audit
        self._registry = registry
        self._artifact_root = Path(artifact_root).resolve()

    @staticmethod
    def _to_dict(model: RegisteredModel) -> dict[str, Any]:
        """Render a registered model without exposing any filesystem path.

        Only ``models:/`` URIs and the package-artifact basename's absence are
        guaranteed here; the internal ``artifact_uri`` (basename) is never
        included in the response.
        """
        uri = model.mlflow_model_uri
        if not (isinstance(uri, str) and uri.startswith(_MODELS_PREFIX)):
            uri = None
        return {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "model_run_id": model.model_run_id,
            "experiment_id": model.experiment_id,
            "mlflow_model_uri": uri,
            "alias": model.alias,
            "deployment_status": model.deployment_status,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
        }

    # -- helpers -----------------------------------------------------------

    def _get_or_raise(self, registered_id: str) -> RegisteredModel:
        model = self._model_repository.get(registered_id)
        if model is None:
            raise RegisteredModelNotFoundError(
                f"registered model {registered_id!r} not found"
            )
        return model

    def _require_recommended_run(self, model_run_id: str) -> tuple:
        """Return ``(model_run, experiment)`` after verifying P0 eligibility."""
        model_run = self._model_repository.get_model_run(model_run_id)
        if model_run is None:
            raise ModelRunNotFoundError(f"model run {model_run_id!r} not found")

        experiment = None
        if model_run.experiment_id:
            experiment = self._experiment_repository.get(model_run.experiment_id)
        if experiment is None:
            raise ExperimentNotEligibleError(
                "model run is not attached to a successful experiment"
            )
        if experiment.status != EXPERIMENT_STATUS_SUCCEEDED:
            raise ExperimentNotEligibleError(
                f"experiment {experiment.id!r} has not succeeded"
            )
        if experiment.recommended_run_id != model_run.mlflow_run_id:
            raise ExperimentNotEligibleError(
                "model run is not the recommended run of its experiment"
            )
        if model_run.model_name != experiment.recommended_model:
            raise ExperimentNotEligibleError(
                "model run does not match the recommended model"
            )
        return model_run, experiment

    def _load_package(self, experiment, model_run) -> tuple[Any, Any]:
        """Load the controlled package artifact and verify it matches the run."""
        snapshot = experiment.training_plan_json if isinstance(
            experiment.training_plan_json, dict
        ) else {}
        artifact_name = snapshot.get("package_artifact")
        if not artifact_name:
            raise ModelRegistryServiceError(
                "experiment has no controlled package artifact to register"
            )
        try:
            pipeline, manifest = package.load_artifact(self._artifact_root, artifact_name)
        except ArtifactError as exc:
            raise ModelRegistryServiceError(f"package artifact is invalid: {exc}") from exc
        if manifest.model_name != model_run.model_name:
            raise ModelRegistryServiceError(
                "package artifact model does not match the model run"
            )
        if manifest.task != experiment.task_type:
            raise ModelRegistryServiceError(
                "package artifact task does not match the experiment"
            )
        return pipeline, manifest

    # -- public API --------------------------------------------------------

    def register(self, model_run_id: str, name: str | None = None) -> dict[str, Any]:
        """Register a successful experiment's recommended run as ``candidate``.

        Duplicate registration of the same run under the same name is idempotent;
        registering it under a different name raises a conflict. The controlled
        package artifact must load and match the run before the MLflow registry is
        touched. Never sets ``champion`` or ``Production``.
        """
        model_run, experiment = self._require_recommended_run(model_run_id)
        name = name or model_run.model_name

        existing = self._model_repository.find_by_model_run(model_run.id)
        if existing:
            same_name = [rm for rm in existing if rm.name == name]
            if same_name:
                return self._to_dict(same_name[0])
            raise RegisterConflictError(
                f"model run {model_run.id!r} is already registered under another name"
            )

        # Confirm the controlled package artifact before mutating the registry.
        self._load_package(experiment, model_run)

        try:
            rv = self._registry.register_run(model_run.mlflow_run_id, name)
        except RegistryError as exc:
            raise ModelRegistryServiceError(str(exc)) from exc

        registered = RegisteredModel(
            id=new_id(),
            model_run_id=model_run.id,
            experiment_id=experiment.id,
            name=name,
            version=rv.version,
            artifact_uri=experiment.training_plan_json.get("package_artifact"),
            mlflow_model_uri=rv.model_uri,
            alias=rv.alias or CANDIDATE_ALIAS,
            approval_status="not_required_demo",
            deployment_status=DEPLOYMENT_STATUS_CANDIDATE,
            deployed=False,
        )
        with self._session_factory.begin() as session:
            registered = self._model_repository.create(registered, session=session)
            self._audit.record(
                actor_type="system",
                action="model.register",
                entity_type="registered_model",
                entity_id=registered.id,
                payload={
                    "registered_model_id": registered.id,
                    "name": name,
                    "version": rv.version,
                    "model_run_id": model_run.id,
                    "experiment_id": experiment.id,
                    "mlflow_model_uri": rv.model_uri,
                    "alias": CANDIDATE_ALIAS,
                    "deployment_status": DEPLOYMENT_STATUS_CANDIDATE,
                },
                session=session,
            )
        return self._to_dict(registered)

    def deploy_demo(self, registered_id: str) -> dict[str, Any]:
        """Deploy one registered model as the demo model (never ``champion``).

        Sets this model to ``demo_deployed`` and demotes every other version of
        the same name to ``candidate``, so exactly one version per name is ever
        deployed. The row update and its audit event commit atomically.
        """
        with self._session_factory.begin() as session:
            model = self._model_repository.get(registered_id, session=session)
            if model is None:
                raise RegisteredModelNotFoundError(
                    f"registered model {registered_id!r} not found"
                )
            model.deployment_status = DEPLOYMENT_STATUS_DEMO_DEPLOYED
            model.deployed = True
            for sibling in self._model_repository.list_by_name(model.name, session=session):
                if sibling.id != model.id:
                    sibling.deployment_status = DEPLOYMENT_STATUS_CANDIDATE
                    sibling.deployed = False
            self._audit.record(
                actor_type="system",
                action="model.deploy_demo",
                entity_type="registered_model",
                entity_id=model.id,
                payload={
                    "registered_model_id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "deployment_status": DEPLOYMENT_STATUS_DEMO_DEPLOYED,
                },
                session=session,
            )
        return self._to_dict(model)

    def get(self, registered_id: str) -> dict[str, Any]:
        """Return a registered model without exposing the artifact basename."""
        return self._to_dict(self._get_or_raise(registered_id))

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return registered-model summaries (no artifact basenames, no paths)."""
        models = self._model_repository.list(limit=limit, offset=offset)
        return [self._to_dict(model) for model in models]
