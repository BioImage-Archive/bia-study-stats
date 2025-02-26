import json
import lzma
from pathlib import Path
from typing import Optional, List
from urllib.parse import quote

import typer
from rich import print
from rich.table import Table
from rich.console import Console


from .s3_size_calculator import Settings, S3SizeCalculator  # Import from your existing module
from .models import BIAStudyStats
from .get_all_bia_accessions import get_all_bia_accessions

app = typer.Typer()

def load_study_stats(stats_file: Path) -> dict[str, BIAStudyStats]:
    """
    Load BIA study statistics from a JSON file and return a dictionary of BIAStudyStats objects.
    
    Args:
        stats_file: Path to the JSON file containing study statistics
        
    Returns:
        dict: Dictionary mapping accession IDs to BIAStudyStats objects
    """
    with open(stats_file) as f:
        stats_data = json.load(f)
    
    return {
        accession: BIAStudyStats(**study_data)
        for accession, study_data in stats_data.items()
    }

@app.command()
def print_accessions(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """
    Print accession IDs and their sizes from a BIA study statistics JSON file.
    """
    # Read the JSON file
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Create a table
    table = Table(title="BIA Study Statistics")
    table.add_column("Accession ID", style="cyan")
    table.add_column("Size (bytes)", justify="right", style="green")
    
    # Add rows to table
    for accession, study_stats in stats.items():
        table.add_row(
            study_stats['accession_id'],
            f"{study_stats['total_size_bytes']:,}"
        )
    
    # Print the table
    console = Console()
    console.print(table)

@app.command()
def merge_df_sizes(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
    df_output: Path = typer.Argument(
        ...,
        help="Path to the file containing sizes from 'df' command",
        exists=True,
    ),
):
    """
    Read sizes from a 'df' command output file and merge them into the BIA study statistics JSON file.
    The input file should be the output of a df command showing disk usage in bytes.
    """
    # Read the JSON file
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Read and parse the df output file
    size_dict = {}
    with open(df_output) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            size_kb, path = parts
            # Extract accession from path (e.g., "./002/S-BIAD1002")
                
            accession = path.split('/S-BIAD')[-1]
            accession = f"S-BIAD{accession}"
            size_bytes = 1024 * int(size_kb)  # Already in bytes, no conversion needed
            size_dict[accession] = size_bytes
    
    # Update sizes in stats
    updates = 0
    for accession, study_stats in stats.items():
        if accession in size_dict:
            study_stats['total_size_bytes'] = size_dict[accession]
            updates += 1
    
    # Save updated JSON
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[green]Updated {updates} studies with size information[/green]")

@app.command()
def merge_s3_cache(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
    s3_cache: Path = typer.Argument(
        ...,
        help="Path to the S3 cache JSON file containing size information",
        exists=True,
    ),
):
    """
    Read sizes from an S3 cache JSON file and merge them into the BIA study statistics JSON file.
    The input file should contain S3 paths and sizes in bytes.
    """
    # Read the stats JSON file
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Read the S3 cache file
    with open(s3_cache) as f:
        s3_sizes = json.load(f)
    
    # Create mapping of accessions to sizes
    size_dict = {}
    for s3_path, size in s3_sizes.items():
        # Extract accession from path (e.g., "biostudies-public:S-BIAD/536/S-BIAD1536")
        if '/S-BIAD' not in s3_path:
            continue
            
        accession = s3_path.split('/S-BIAD')[-1]
        accession = f"S-BIAD{accession}"
        size_dict[accession] = int(size)  # Ensure size is integer
    
    # Update sizes in stats
    updates = 0
    for accession, study_stats in stats.items():
        if accession in size_dict:
            study_stats['total_size_bytes'] = size_dict[accession]
            updates += 1
    
    # Save updated JSON
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[green]Updated {updates} studies with size information from S3 cache[/green]")

@app.command()
def update_from_fire(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
    failed_log: Optional[Path] = typer.Option(
        None,
        help="Path to log file where failed updates will be recorded",
    ),
):
    """
    Update studies with zero size by fetching their sizes from S3/FIRE storage.
    Requires proper S3 configuration in .env file.
    """
    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        print(f"[red]Error loading S3 configuration: {e}[/red]")
        print("[yellow]Please ensure your .env file exists with required configuration:[/yellow]")
        print("S3_BUCKET=your-bucket-name")
        print("S3_ENDPOINT=https://your-endpoint.com  # Optional")
        print("AWS_PROFILE=your-profile  # Optional")
        raise typer.Exit(1)

    # Initialize calculator
    calculator = S3SizeCalculator(settings)
    
    # Read the stats file
    with open(stats_file) as f:
        stats = json.load(f)
    
    # Find studies with zero size
    zero_size_studies = [
        accession for accession, study in stats.items()
        if study['total_size_bytes'] == 0
    ]
    
    if not zero_size_studies:
        print("[green]No studies found with zero size - nothing to update[/green]")
        return
    
    print(f"[yellow]Found {len(zero_size_studies)} studies with zero size[/yellow]")
    
    # Track failed updates
    failed_updates = []
    
    # Update sizes
    updates = 0
    with typer.progressbar(zero_size_studies, label="Updating sizes") as progress:
        for accession in progress:
            try:
                # Convert accession to S3 prefix
                s3_prefix = calculator.translate_identifier_to_key(accession)
                size = calculator.calculate_prefix_size(s3_prefix)
                
                if size > 0:
                    stats[accession]['total_size_bytes'] = size
                    updates += 1
                else:
                    print(f"[red]Failed to get size for {accession}[/red]")
                    failed_updates.append((accession, "Zero size returned"))
            except Exception as e:
                print(f"[red]Error processing {accession}: {e}[/red]")
                failed_updates.append((accession, str(e)))
    
    # Save updated stats
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Log failed updates if requested
    if failed_log and failed_updates:
        with open(failed_log, 'w') as f:
            for accession, error in failed_updates:
                f.write(f"{accession}\t{error}\n")
        print(f"[yellow]Logged {len(failed_updates)} failed updates to {failed_log}[/yellow]")
    
    print(f"[green]Updated sizes for {updates} studies[/green]")

@app.command()
def summarize(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """
    Summarize the statistics file, showing total number of accessions and how many have sizes.
    """
    # Read the JSON file
    with open(stats_file) as f:
        stats = json.load(f)
    
    total_accessions = len(stats)
    accessions_with_size = sum(1 for study in stats.values() if study['total_size_bytes'] > 0)
    total_size_bytes = sum(study['total_size_bytes'] for study in stats.values())
    
    # Create a table
    table = Table(title="BIA Study Statistics Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Total Accessions", str(total_accessions))
    table.add_row("Accessions with Size", str(accessions_with_size))
    table.add_row("Accessions without Size", str(total_accessions - accessions_with_size))
    table.add_row("Total Size", f"{total_size_bytes:,} bytes")
    
    # Print the table
    console = Console()
    console.print(table)

@app.command()
def data_added_after(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
    date: str = typer.Argument(
        ...,
        help="Date in YYYY-MM-DD format to check data added after",
    ),
):
    """
    Summarize the total volume of data added after a specific date.
    """
    # Parse the input date
    try:
        from datetime import datetime
        cutoff_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        print("[red]Invalid date format. Please use YYYY-MM-DD[/red]")
        raise typer.Exit(1)

    # Load study stats
    studies = load_study_stats(stats_file)
    
    # Calculate totals
    total_size = 0
    studies_count = 0
    
    for study in studies.values():
        if study.release_date > cutoff_date:
            total_size += study.total_size_bytes
            studies_count += 1
    
    # Create a table
    table = Table(title=f"BIA Data Added After {date}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Number of Studies", str(studies_count))
    table.add_row("Total Size", f"{total_size:,} bytes")
    
    # Print the table
    console = Console()
    console.print(table)

@app.command()
def plot_cumulative_size(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """Plot cumulative size of total data over time as quarterly bars since 2019."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    # Load study stats
    studies = load_study_stats(stats_file)
    
    # Create DataFrame with release dates and sizes
    df = pd.DataFrame([
        {
            'release_date': pd.Timestamp(study.release_date),  # Convert date to Timestamp
            'size_bytes': study.total_size_bytes
        }
        for study in studies.values()
    ])
    
    # Filter for dates since 2019
    df = df[df['release_date'] >= pd.Timestamp('2019-01-01')]
    
    # Group by quarter and sum sizes
    df['quarter'] = pd.PeriodIndex(df['release_date'], freq='Q')
    quarterly_data = df.groupby('quarter')['size_bytes'].sum()
    
    # Calculate cumulative sum in TB
    quarterly_data_cumsum_tb = quarterly_data.cumsum() / (1024**4)
    
    # Create the plot
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    
    # Create bar plot
    ax = quarterly_data_cumsum_tb.plot(kind='bar')
    
    # Customize the plot
    plt.title('Cumulative Data Size in BioImage Archive by Quarter (Since 2019)', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Total Data (TB)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for i, v in enumerate(quarterly_data_cumsum_tb):
        ax.text(i, v, f'{v:.1f}TB', ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('quarterly_cumulative_size.png')
    print(f"Plot saved as quarterly_cumulative_size.png")

@app.command()
def print_ebi_stats(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """Print monthly cumulative size statistics in EBI format (YYYYMM total_bytes)."""
    import pandas as pd
    
    # Load study stats
    studies = load_study_stats(stats_file)
    
    # Create DataFrame with release dates and sizes
    df = pd.DataFrame([
        {
            'release_date': pd.Timestamp(study.release_date),
            'size_bytes': study.total_size_bytes
        }
        for study in studies.values()
    ])
    
    # Sort by date
    df = df.sort_values('release_date')
    
    # Group by year-month and calculate cumulative sum
    df['yearmonth'] = df['release_date'].dt.strftime('%Y%m')
    monthly_data = df.groupby('yearmonth')['size_bytes'].sum().cumsum()
    
    # Print in required format
    for yearmonth, total_bytes in monthly_data.items():
        print(f"{yearmonth} {int(total_bytes)}")

@app.command()
def plot_cumulative_entries(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """Plot cumulative number of studies over time as quarterly bars since 2019."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load study stats
    studies = load_study_stats(stats_file)
    
    # Create DataFrame with release dates
    df = pd.DataFrame([
        {
            'release_date': pd.Timestamp(study.release_date),
            'count': 1  # Each study counts as 1
        }
        for study in studies.values()
    ])
    
    # Filter for dates since 2019
    df = df[df['release_date'] >= pd.Timestamp('2019-01-01')]
    
    # Group by quarter and count studies
    df['quarter'] = pd.PeriodIndex(df['release_date'], freq='Q')
    quarterly_data = df.groupby('quarter')['count'].sum()
    
    # Calculate cumulative sum
    quarterly_data_cumsum = quarterly_data.cumsum()
    
    # Create the plot
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    
    # Create bar plot
    ax = quarterly_data_cumsum.plot(kind='bar')
    
    # Customize the plot
    plt.title('Cumulative Number of Studies in BioImage Archive by Quarter (Since 2019)', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Total Number of Studies', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for i, v in enumerate(quarterly_data_cumsum):
        ax.text(i, v, f'{int(v):,}', ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('quarterly_cumulative_entries.png')
    print(f"Plot saved as quarterly_cumulative_entries.png")

@app.command()
def bfftree_for_biostudies_entry(
    accession_id: str = typer.Argument(..., help="BioStudies accession ID (e.g. S-BIAD1654)"),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Output path for the BFFTree (defaults to bfftrees/biostudies/xz/{accession_id}.pb.xz)",
    ),
):
    """
    Create a BFFTree from a BioStudies entry's S3 data and save it as a compressed protobuf file.
    """
    from .web_utils import extract_ftp_link
    
    # Set default output path if none provided
    if output_path is None:
        output_path = Path(f"bfftrees/biostudies/xz/{accession_id}.pb.xz")
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get FTP link for the accession
        ftp_link = extract_ftp_link(accession_id)
        
        # Extract the relevant part of the path and construct S3 prefix
        # e.g. from ftp://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/654/S-BIAD1654
        # to S-BIAD/654/S-BIAD1654/Files
        path_parts = ftp_link.split('/fire/')[-1].split('/')
        prefix = f"{'/'.join(path_parts)}/Files"
        
        # URL encode each part of the prefix separately, keeping slashes intact
        encoded_prefix = '/'.join(quote(part, safe=' ') for part in prefix.split('/'))
        
        # Call bfftree_from_s3_prefix with encoded prefix
        bfftree_from_s3_prefix(
            prefix=prefix,
            output_path=output_path,
            bucket="biostudies-public"
        )
        
    except ValueError as e:
        print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)

def bfftree_for_empiar_entry(
    empiar_id: str = typer.Argument(..., help="EMPIAR ID (e.g. EMPIAR-10473)"),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Output path for the BFFTree (defaults to bfftrees/xz/EMPIAR-{id}.pb.xz)",
    ),
):
    """
    Create a BFFTree from an EMPIAR entry's S3 data and save it as a compressed protobuf file.
    """
    # Extract numeric ID from EMPIAR ID
    if not empiar_id.startswith("EMPIAR-"):
        print("[red]EMPIAR ID must start with 'EMPIAR-'[/red]")
        raise typer.Exit(1)
    numeric_id = empiar_id.split("-")[1]
    
    # Set default output path if none provided
    if output_path is None:
        output_path = Path(f"bfftrees/empiar/xz/{empiar_id}.pb.xz")
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Construct S3 prefix for EMPIAR data
    prefix = f"world_availability/{numeric_id}/data"
    
    # Call bfftree_from_s3_prefix with EMPIAR-specific settings
    bfftree_from_s3_prefix(
        prefix=prefix,
        output_path=output_path,
        bucket="imaging-public"
    )

@app.command()
def bfftree_from_s3_prefix(
    prefix: str = typer.Argument(..., help="S3 prefix to process"),
    output_path: Path = typer.Argument(..., help="Output path for the BFFTree (.pb or .pb.xz)"),
    bucket: str = typer.Option("biostudies-public", help="S3 bucket name"),
):
    """
    Create a BFFTree from an S3 prefix and save it as a protobuf file.
    Supports optional compression if output path ends with .xz
    """
    from .s3_utils import S3Settings, s3_prefix_to_bfftree
    
    # Load settings
    settings = S3Settings()
    
    try:
        # Generate the tree
        print(f"[yellow]Generating BFFTree from s3://{bucket}/{prefix}...[/yellow]")
        tree = s3_prefix_to_bfftree(settings, bucket, prefix, strip_prefix=True)
        
        # Save as compressed protobuf if output ends with .xz
        if str(output_path).endswith('.xz'):
            with lzma.open(output_path, 'wb') as f:
                f.write(tree.to_proto().SerializeToString())
        else:
            tree.save_to_proto_file(output_path)
            
        print(f"[green]BFFTree saved to {output_path}[/green]")
        
    except Exception as e:
        print(f"[red]Error generating BFFTree: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def generate_all_empiar_bfftrees(
    force: bool = typer.Option(False, help="Force regeneration of existing BFFTrees")
):
    """
    Generate BFFTrees for all EMPIAR entries listed in resources/all_empiar_entry_identifiers.json.
    Skip entries that already have BFFTrees unless --force is used.
    """
    # Load the list of EMPIAR IDs
    try:
        with open("resources/all_empiar_entry_identifiers.json") as f:
            empiar_ids: List[str] = json.load(f)
    except FileNotFoundError:
        print("[red]Error: resources/all_empiar_entry_identifiers.json not found[/red]")
        raise typer.Exit(1)
    
    print(f"[yellow]Found {len(empiar_ids)} EMPIAR entries to process[/yellow]")
    
    # Process each EMPIAR ID
    for empiar_id in empiar_ids:
        output_path = Path(f"bfftrees/empiar/xz/{empiar_id}.pb.xz")
        
        # Skip if file exists and not forcing regeneration
        if output_path.exists() and not force:
            print(f"[blue]Skipping {empiar_id} - BFFTree already exists at {output_path}[/blue]")
            continue
        
        print(f"[yellow]Processing {empiar_id}...[/yellow]")
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Call with explicit output path to avoid creating it twice
            bfftree_for_empiar_entry(
                empiar_id=empiar_id,
                output_path=output_path
            )
            print(f"[green]Successfully generated BFFTree for {empiar_id}[/green]")
        except Exception as e:
            print(f"[red]Error generating BFFTree for {empiar_id}: {e}[/red]")
            continue

@app.command()
def analyze_bfftrees(
    directory: Path = typer.Argument(..., help="Directory containing .pb.xz BFFTree files"),
    limit: Optional[int] = typer.Option(None, help="Limit analysis to first N files"),
):
    """
    Analyze multiple BFFTree files in a directory and produce a summary of sizes and file types.
    """
    import lzma
    from pathlib import Path
    from bfftree.bettertree_pb2 import RadixTreeNode as ProtoNode
    from bfftree.tree import RadixTreeNode
    
    # Find all .pb.xz files
    tree_files = list(directory.glob("**/*.pb.xz"))
    if limit:
        tree_files = tree_files[:limit]
    
    if not tree_files:
        print("[red]No .pb.xz files found in directory[/red]")
        raise typer.Exit(1)
    
    print(f"[yellow]Analyzing {len(tree_files)} BFFTree files...[/yellow]")
    
    # Initialize counters
    total_size = 0
    total_files = 0
    ext_sizes = {}
    ext_counts = {}
    
    # Process each file
    with typer.progressbar(tree_files) as progress:
        for tree_file in progress:
            try:
                # Load the tree
                proto_node = ProtoNode()
                with lzma.open(tree_file, 'rb') as f:
                    proto_node.ParseFromString(f.read())
                tree = RadixTreeNode.load_from_proto_file_proto(proto_node)
                
                # Get paths and update statistics
                paths = tree.get_all_paths()
                total_files += len(paths)
                
                for path, size in paths:
                    total_size += size
                    ext = Path(path).suffix.lower()
                    if not ext:
                        ext = "(no extension)"
                    ext_sizes[ext] = ext_sizes.get(ext, 0) + size
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    
            except Exception as e:
                print(f"[red]Error processing {tree_file}: {e}[/red]")
                continue
    
    # Format size for human readability
    def human_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    # Create summary table
    console = Console()
    console.print(f"\n[bold]Summary of {len(tree_files)} BFFTree files:[/bold]\n")
    
    summary_table = Table()
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")
    
    summary_table.add_row("Total Files", f"{total_files:,}")
    summary_table.add_row("Total Size", f"{human_size(total_size)} ({total_size:,} bytes)")
    
    console.print(summary_table)
    
    # Create extension breakdown table
    if ext_sizes:
        ext_table = Table(title="\nBreakdown by File Extension")
        ext_table.add_column("Extension", style="cyan")
        ext_table.add_column("Count", justify="right", style="green")
        ext_table.add_column("Total Size", justify="right", style="green")
        ext_table.add_column("% of Total", justify="right", style="magenta")
        
        # Add rows sorted by size
        for ext, size in sorted(ext_sizes.items(), key=lambda x: x[1], reverse=True):
            count = ext_counts[ext]
            percentage = (size / total_size) * 100
            ext_table.add_row(
                ext,
                f"{count:,}",
                human_size(size),
                f"{percentage:.1f}%"
            )
        
        console.print("\n")
        console.print(ext_table)

@app.command()
def yearly_stats(
    stats_file: Path = typer.Argument(
        ...,
        help="Path to the JSON file containing BIA study statistics",
        exists=True,
    ),
):
    """Summarize cumulative number of studies and total data volume for each year 2019-2024."""
    import pandas as pd
    
    # Load study stats
    studies = load_study_stats(stats_file)
    
    # Create DataFrame with release dates and sizes
    df = pd.DataFrame([
        {
            'release_date': pd.Timestamp(study.release_date),
            'size_bytes': study.total_size_bytes,
            'count': 1  # Each study counts as 1
        }
        for study in studies.values()
    ])
    
    # Filter for dates since 2019
    df = df[df['release_date'] >= pd.Timestamp('2019-01-01')]
    
    # Create table
    table = Table(title="BIA Yearly Statistics (Cumulative)")
    table.add_column("Year", style="cyan")
    table.add_column("Total Studies", justify="right", style="green")
    table.add_column("Total Data", justify="right", style="green")
    
    # Calculate cumulative stats for each year
    for year in range(2019, 2025):
        year_end = pd.Timestamp(f"{year}-12-31")
        year_data = df[df['release_date'] <= year_end]
        
        total_studies = len(year_data)
        total_bytes = year_data['size_bytes'].sum()
        
        # Convert bytes to more readable format
        if total_bytes >= 1024**4:  # TB
            size_str = f"{total_bytes/1024**4:.1f} TB"
        elif total_bytes >= 1024**3:  # GB
            size_str = f"{total_bytes/1024**3:.1f} GB"
        else:
            size_str = f"{total_bytes:,} bytes"
        
        table.add_row(
            str(year),
            f"{total_studies:,}",
            size_str
        )
    
    # Print the table
    console = Console()
    console.print(table)

@app.command()
def generate_all_biostudies_bfftrees(
    force: bool = typer.Option(False, help="Force regeneration of existing BFFTrees")
):
    """
    Generate BFFTrees for all BioStudies entries from BioImage Archive.
    Skip entries that already have BFFTrees unless --force is used.
    """
    print("[yellow]Fetching all BIA accessions...[/yellow]")
    studies = get_all_bia_accessions()
    print(f"[yellow]Found {len(studies)} BIA entries to process[/yellow]")
    
    # Process each accession
    for study in studies:
        accession_id = study.accession
        output_path = Path(f"bfftrees/biostudies/xz/{accession_id}.pb.xz")
        
        # Skip if file exists and not forcing regeneration
        if output_path.exists() and not force:
            print(f"[blue]Skipping {accession_id} - BFFTree already exists at {output_path}[/blue]")
            continue
        
        print(f"[yellow]Processing {accession_id}...[/yellow]")
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Call with explicit output path to avoid creating it twice
            bfftree_for_biostudies_entry(
                accession_id=accession_id,
                output_path=output_path
            )
            print(f"[green]Successfully generated BFFTree for {accession_id}[/green]")
        except Exception as e:
            print(f"[red]Error generating BFFTree for {accession_id}: {e}[/red]")
            continue

if __name__ == "__main__":
    app() 
