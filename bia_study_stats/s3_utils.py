import boto3  # type: ignore
import botocore  # type: ignore
import typer
from typing import Optional, List
from urllib.parse import quote
from pydantic import BaseModel
from bfftree.tree import RadixTreeNode
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

class S3Settings(BaseSettings):
    """S3 configuration settings"""
    s3_endpoint: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

def calculate_prefix_size(settings: S3Settings, bucket: str, prefix: str) -> int:
    """
    Calculate the total size of all objects under a prefix in an S3 bucket.
    Handles recursive listing manually for S3 services that don't support it.
    
    Args:
        settings: S3Settings instance with configuration
        prefix: The prefix to calculate size for
        
    Returns:
        int: Total size in bytes
    """
    # Initialize S3 client with unsigned requests (equivalent to --no-sign-request)
    client_kwargs = {'config': botocore.client.Config(signature_version=botocore.UNSIGNED)}
    if settings.s3_endpoint:
        client_kwargs['endpoint_url'] = settings.s3_endpoint
    s3_client = boto3.client('s3', **client_kwargs)
    
    total_size = 0
    prefixes_to_process = [prefix]
    
    try:
        # Print equivalent AWS CLI command
        cli_command = f"aws s3 ls s3://{bucket}/{prefix} --recursive --no-sign-request"
        if settings.s3_endpoint:
            cli_command += f" --endpoint-url {settings.s3_endpoint}"
        
        typer.secho(f"\nEquivalent AWS CLI command:", fg=typer.colors.BLUE)
        typer.secho(cli_command, fg=typer.colors.BLUE)
        
        while prefixes_to_process:
            current_prefix = prefixes_to_process.pop(0)
            # URL encode each part of the prefix separately, keeping slashes and spaces intact
            # Handle non-ASCII characters by encoding to UTF-8 first
            encoded_prefix = '/'.join(quote(part.encode('utf-8').decode('utf-8'), safe=' ') for part in current_prefix.split('/'))
            paginator = s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket, Prefix=encoded_prefix, Delimiter='/'):
                # Add sizes of objects at current level
                if 'Contents' in page:
                    total_size += sum(obj['Size'] for obj in page['Contents'])
                
                # Queue up any common prefixes for processing
                if 'CommonPrefixes' in page:
                    prefixes_to_process.extend(
                        p['Prefix'] for p in page['CommonPrefixes']
                    )
        
        return total_size
        
    except s3_client.exceptions.ClientError as e:
        typer.secho(f"Error accessing S3: {e}", fg=typer.colors.RED)
        return -1


class FileEntry(BaseModel):
    path: str
    size: int


def s3_prefix_to_bfftree(settings: S3Settings, bucket: str, prefix: str, strip_prefix: bool = False) -> 'RadixTreeNode':
    """
    Create a RadixTree from an S3 prefix by recursively listing all objects.
    
    Args:
        settings: S3Settings instance with configuration
        bucket: The S3 bucket name
        prefix: The prefix to process
        strip_prefix: If True, remove the prefix from stored paths
        
    Returns:
        RadixTree representation of the S3 prefix contents
    """
    # Initialize S3 client with unsigned requests
    client_kwargs = {'config': botocore.client.Config(signature_version=botocore.UNSIGNED)}
    if settings.s3_endpoint:
        client_kwargs['endpoint_url'] = settings.s3_endpoint
    s3_client = boto3.client('s3', **client_kwargs)
    
    entries: List[FileEntry] = []
    prefixes_to_process = [prefix]
    
    try:
        while prefixes_to_process:
            current_prefix = prefixes_to_process.pop(0)
            # URL encode each part of the prefix separately, keeping slashes and spaces intact
            # Handle non-ASCII characters by encoding to UTF-8 first
            encoded_prefix = '/'.join(quote(part.encode('utf-8').decode('utf-8'), safe=' ') for part in current_prefix.split('/'))
            paginator = s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket, Prefix=encoded_prefix, Delimiter='/'):
                # Process objects at current level
                if 'Contents' in page:
                    for obj in page['Contents']:
                        path = obj['Key']
                        if strip_prefix and path.startswith(prefix):
                            path = path[len(prefix):].lstrip('/')
                        entries.append(FileEntry(
                            path=path,
                            size=obj['Size']
                        ))
                
                # Queue up any common prefixes for processing
                if 'CommonPrefixes' in page:
                    prefixes_to_process.extend(
                        p['Prefix'] for p in page['CommonPrefixes']
                    )
        
        # Build and return the radix tree
        root = RadixTreeNode()
        for entry in entries:
            root.insert(entry.path, entry.size)
        return root
        
    except s3_client.exceptions.ClientError as e:
        typer.secho(f"Error accessing S3: {e}", fg=typer.colors.RED)
        raise
