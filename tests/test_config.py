"""Configuration tests (pydantic-settings)."""

from pydantic import SecretStr

from maintai.config import Settings, get_settings


def test_init_args_override_defaults():
    s = Settings(project_name="custom", random_seed=7)
    assert s.project_name == "custom"
    assert s.random_seed == 7


def test_empty_llm_strings_normalize_to_none():
    s = Settings(llm_api_key="", llm_base_url="   ", llm_model="")
    assert s.llm_api_key is None
    assert s.llm_base_url is None
    assert s.llm_model is None


def test_llm_provider_defaults_to_mock():
    assert Settings().llm_provider == "mock"


def test_llm_key_is_masked():
    settings = Settings(llm_api_key="sk-test-placeholder")
    assert isinstance(settings.llm_api_key, SecretStr)
    assert "sk-test-placeholder" not in repr(settings)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_data_and_split_yaml_are_loaded():
    settings = Settings()
    assert settings.data.missing_warning_rate == 0.05
    assert settings.data.flatline_min_length == 10
    assert settings.split.test_size == 0.20
    assert settings.split.validation_size == 0.20
    assert settings.ml.minimum_recall == 0.80
    assert settings.ml.n_jobs == 1
