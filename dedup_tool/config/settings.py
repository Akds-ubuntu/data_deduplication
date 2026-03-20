from pathlib import Path
from typing import Literal
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DedupConfig(BaseSettings):
    """Main configuration for deduplication."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    algorithm: Literal["minhash", "semantic", "simhash", "semantic"] = Field(
        default="minhash", description="Deduplication algorithm to use"
    )

    input_file: str = Field(
        default="", description="Path to input file or directory with texts"
    )
    output_path: str = Field(default="results.json", description="Path to save results")

    num_perm: int = Field(default=128, description="Number of permutations for MinHash")
    ngram_size: int = Field(default=5, description="N-gram size for shingling")

    bands: int = Field(default=16, description="Number of bands in LSH")

    threshold: float | int = Field(
        default=0.5, ge=0.0, le=64.0, description="Similarity threshold (0.0-64.0)"
    )
    verify_ast: bool = Field(default=False)
    use_semantic_lsh: bool = Field(default=False)
    threshold_semantic_lsh: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)"
    )

    model_name: str | None = Field(
        default="BAAI/bge-small-en-v1.5",
    )

    use_bloom: bool = Field(default=False, description="Use Bloom filter optimization")
    debug: bool = Field(default=False, description="Enable debug logging")
    text_field: str = Field(default="")

    @classmethod
    def from_yaml(cls, yaml_path: Path = Path("config.yaml")):
        """Load configuration from YAML file."""
        if not yaml_path.exists():
            return cls()

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)
