import typer
import boto3 # type: ignore
import json
from typing import Dict, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore

app = typer.Typer()

class Settings(BaseSettings):
    """S3 configuration settings"""
    s3_bucket: str
    s3_endpoint: Optional[str] = None
    aws_profile: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

class S3SizeCalculator:
    def __init__(self, settings: Settings, cache_file: Path = Path("s3_size_cache.json")):
        """
        Initialize the calculator with boto3 client and cache file path.
        
        Args:
            settings: Pydantic Settings instance with S3 configuration
            cache_file: Path to the JSON cache file
        """
        self.settings = settings
        
        # Initialize S3 client
        session = boto3.Session(profile_name=settings.aws_profile)
        self.s3_client = session.client('s3', endpoint_url=settings.s3_endpoint) if settings.s3_endpoint else session.client('s3')
        
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, int]:
        """Load the cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                return json.loads(self.cache_file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_cache(self):
        """Save the current cache to disk."""
        self.cache_file.write_text(json.dumps(self.cache, indent=2))
    
    def translate_identifier_to_key(self, identifier: str) -> str:
        """
        Translate an identifier to an S3 key prefix.
        The identifier format is PREFIX followed by up to 4 digits (NUMBER).
        The path is constructed as: PREFIX/DIGITS/IDENTIFIER where:
        - For S-BSST studies (101-502): PREFIX/S-BSSTxxx413/IDENTIFIER
        - For all others: PREFIX/last_3_digits/IDENTIFIER
        
        Examples:
            S-BIAD1536 -> S-BIAD/536/S-BIAD1536
            S-BIAD42 -> S-BIAD/042/S-BIAD42
            S-BIAD9 -> S-BIAD/009/S-BIAD9
            S-BSST101 -> S-BSST/S-BSSTxxx413/S-BSST101
        
        Args:
            identifier: The identifier to translate (e.g., "S-BIAD1536")
            
        Returns:
            str: The corresponding S3 key prefix
        """
        try:
            # Special case for S-BSST studies
            if identifier.startswith('S-BSST'):
                # Find where the number starts
                for i, char in enumerate(identifier):
                    if char.isdigit():
                        number = int(identifier[i:])
                        if 101 <= number <= 502:
                            return f"S-BSST/S-BSSTxxx{number}/{identifier}"
                        break
            
            # Regular case - same as before
            for i, char in enumerate(identifier):
                if char.isdigit():
                    prefix = identifier[:i]
                    number = identifier[i:]
                    break
            else:
                raise ValueError(f"No number found in identifier: {identifier}")
            
            # Pad the number to 5 digits
            padded_number = number.zfill(5)
            # Get the last 3 digits for the middle path
            middle_path = padded_number[-3:]
            
            # Construct the path
            return f"{prefix}/{middle_path}/{identifier}"
            
        except Exception as e:
            raise ValueError(f"Invalid identifier format: {identifier}. Expected format like PREFIX followed by number") from e
    
    def calculate_prefix_size(self, prefix: str, force_recalculate: bool = False) -> int:
        """
        Calculate the total size of all objects under a prefix in an S3 bucket.
        Uses caching to avoid recalculating unless forced.
        Handles recursive listing manually for S3 services that don't support it.
        
        Args:
            prefix: The prefix to calculate size for
            force_recalculate: If True, ignore cache and recalculate
            
        Returns:
            int: Total size in bytes
        """
        cache_key = f"{self.settings.s3_bucket}:{prefix}"
        
        if not force_recalculate and cache_key in self.cache:
            return self.cache[cache_key]
        
        total_size = 0
        prefixes_to_process = [prefix]
        
        try:
            # Print equivalent AWS CLI command
            cli_command = f"aws s3 ls s3://{self.settings.s3_bucket}/{prefix} --recursive"
            if self.settings.s3_endpoint:
                cli_command += f" --endpoint-url {self.settings.s3_endpoint}"
            if self.settings.aws_profile:
                cli_command += f" --profile {self.settings.aws_profile}"
            
            typer.secho(f"\nEquivalent AWS CLI command:", fg=typer.colors.BLUE)
            typer.secho(cli_command, fg=typer.colors.BLUE)
            
            while prefixes_to_process:
                current_prefix = prefixes_to_process.pop(0)
                paginator = self.s3_client.get_paginator('list_objects_v2')
                
                for page in paginator.paginate(Bucket=self.settings.s3_bucket, Prefix=current_prefix, Delimiter='/'):
                    # Add sizes of objects at current level
                    if 'Contents' in page:
                        total_size += sum(obj['Size'] for obj in page['Contents'])
                    
                    # Queue up any common prefixes for processing
                    if 'CommonPrefixes' in page:
                        prefixes_to_process.extend(
                            p['Prefix'] for p in page['CommonPrefixes']
                        )
            
            self.cache[cache_key] = total_size
            self._save_cache()
            return total_size
            
        except self.s3_client.exceptions.ClientError as e:
            typer.secho(f"Error accessing S3: {e}", fg=typer.colors.RED)
            return -1

@app.command()
def calculate_sizes(
    input_file: Path = typer.Argument(..., help="File containing newline-delimited identifiers"),
    cache_file: Path = typer.Option(
        Path("s3_size_cache.json"),
        "--cache-file",
        "-c",
        help="Cache file path"
    ),
    force_recalculate: bool = typer.Option(
        False,
        "--force-recalculate",
        "-f",
        help="Force recalculation ignoring cache"
    )
):
    """
    Calculate total size of S3 objects under prefixes derived from identifiers.
    Results are cached to avoid unnecessary recalculation.
    
    Requires a .env file with the following variables:
    - S3_BUCKET (required): The S3 bucket name
    - S3_ENDPOINT (optional): Custom S3 endpoint URL
    - AWS_PROFILE (optional): AWS profile name to use
    """
    try:
        settings = Settings()
    except Exception as e:
        typer.secho(
            f"Error loading configuration: {e}\n\n"
            "Please ensure your .env file exists with the required configuration:\n\n"
            "S3_BUCKET=your-bucket-name\n"
            "S3_ENDPOINT=https://your-endpoint.com  # Optional\n"
            "AWS_PROFILE=your-profile  # Optional",
            fg=typer.colors.RED
        )
        raise typer.Exit(1)
    
    calculator = S3SizeCalculator(settings, cache_file)
    
    try:
        identifiers = input_file.read_text().splitlines()
        identifiers = [id.strip() for id in identifiers if id.strip()]
        
        with typer.progressbar(identifiers, label="Processing identifiers") as progress:
            for identifier in progress:
                s3_prefix = calculator.translate_identifier_to_key(identifier)
                size = calculator.calculate_prefix_size(
                    s3_prefix, force_recalculate
                )
                
                if size >= 0:
                    typer.secho(
                        f"{identifier}: {size:,} bytes",
                        fg=typer.colors.GREEN
                    )
                else:
                    typer.secho(
                        f"{identifier}: Error calculating size",
                        fg=typer.colors.RED
                    )
                
    except FileNotFoundError:
        typer.secho(f"Error: Input file '{input_file}' not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)

if __name__ == '__main__':
    app()
