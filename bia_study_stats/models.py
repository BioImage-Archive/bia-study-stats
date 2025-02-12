from datetime import date
from pydantic import BaseModel, Field

class BIAStudyStats(BaseModel):
    """
    Statistics for a BioImage Archive study.
    """
    accession_id: str = Field(
        description="The BioImage Archive accession ID (e.g., S-BIAD1536)",
    )
    
    number_of_files: int = Field(
        description="Total number of files in the study",
        ge=0
    )
    
    total_size_bytes: int = Field(
        description="Total size of all files in bytes",
        ge=0
    )
    
    release_date: date = Field(
        description="Date when the study was publicly released"
    )
    
    title: str = Field(
        description="Title of the study"
    ) 