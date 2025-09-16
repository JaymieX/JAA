import toml
from pathlib import Path

class Config:
    def __init__(self, config_path="config.toml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._init_directories()

    def _load_config(self):
        """Load configuration from TOML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return toml.load(f)

    def _init_directories(self):
        """Create necessary directories if they don't exist"""
        data_config = self.config.get("data", {})

        # Create base data directory
        base_dir = Path(data_config.get("base_dir", "data"))
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create raw data directory
        raw_dir = Path(data_config.get("raw_dir", "data/raw"))
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized directories: {base_dir}, {raw_dir}")

    @property
    def data_base_dir(self):
        """Get base data directory path"""
        return Path(self.config["data"]["base_dir"])

    @property
    def data_raw_dir(self):
        """Get raw data directory path"""
        return Path(self.config["data"]["raw_dir"])

    def get_raw_file_path(self, filename):
        """Get full path for a file in the raw data directory"""
        return self.data_raw_dir / filename

# Global config instance
config = Config()