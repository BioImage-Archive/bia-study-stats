import json
import math
import datetime
from typing import List

import requests
import pandas as pd
from pydantic import BaseModel
from models import BIAStudyStats

search_url = "https://www.ebi.ac.uk/biostudies/api/v1/BioImages/search"

class StudyResult(BaseModel):
    accession: str
    title: str
    author: str
    links: int
    files: int
    release_date: datetime.date
    views: int
    isPublic: bool
        
class QueryResult(BaseModel):
    page: int
    pageSize: int
    totalHits: int
    isTotalHitsExact: bool
    sortBy: str
    sortOrder: str
    hits: List[StudyResult]

def get_hits_by_page(page, page_size):
    r = requests.get(search_url, params={"accession": "S-B*", "pageSize": page_size, "page": page})
    qr = QueryResult.parse_raw(r.content)
    return qr.hits

def get_all_bia_accessions():
    page_size = 100
    r = requests.get(search_url, params={"accession": "S-B*", "pageSize": page_size})
    qr = QueryResult.parse_raw(r.content)    
    n_calls = math.ceil(qr.totalHits / page_size)
    
    hits_list_of_lists = [get_hits_by_page(p+1, page_size) for p in range(n_calls)]
    all_studies = sum(hits_list_of_lists, [])
    
    return all_studies


def main():
    all_studies = get_all_bia_accessions()
    
    # Create dictionary with accession IDs as keys
    stats_dict = {}
    for study in all_studies:
            
        try:
            # Create BIAStudyStats object
            stats = BIAStudyStats(
                accession_id=study.accession,
                number_of_files=study.files,
                total_size_bytes=0,  # We'll handle sizes separately
                release_date=study.release_date,
                title=study.title
            )
            # Convert to dict and ensure release_date is string
            stats_dict[study.accession] = stats.model_dump(mode='json')
        except ValueError as e:
            print(f"Skipping {study.accession}: {e}")
            continue
    
    # Write to JSON file
    with open('bia_study_stats.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)


if __name__ == "__main__":
    main()