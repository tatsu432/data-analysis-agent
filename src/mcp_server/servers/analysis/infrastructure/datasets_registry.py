"""Dataset registry for MCP data analysis tools."""

import os
from pathlib import Path
from typing import Any, Dict

import dotenv

# Calculate PROJECT_ROOT: go up 6 levels from this file
# src/mcp_server/servers/analysis/infrastructure/datasets_registry.py -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
dotenv.load_dotenv()

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

JAMDAS_PATIENT_DATA_DESCRIPTION = """The patient data for each product with information about the at risk patients 
and ddi prescriptions. This dataset only covers GP (the medical institution whose number of beds is lower than 20).
 It does not have the information about each prefecture's patinet but overall patient."""

JPM_PATIENT_DATA_DESCRIPTION = """The patient data for each product with information about GP (the medical 
institution whose number of beds is lower than 20) or HP (the medical institution whose number of beds is higher than 20).
It does not have the information about each prefecture's patinet but overall patient."""

COVID_NEW_CASES_DAILY_DESCRIPTION = (
    "COVID-19 newly confirmed cases daily data for Japanese prefectures (local CSV)."
)

MR_ACTIVITY_DATA_DESCRIPTION = """MR (Medical Representative) activity data by prefecture, month, and HP/GP type. 
Contains detailing visits (float), number of emails sent (int), and number of seminars hosted (int) for each 
prefecture-month-HP/GP combination. Covers all 47 prefectures of Japan from 2023-04 to 2025-09. 
HP (Hospitals with 20+ beds) typically have higher activity levels than GP (General Practices with <20 beds)."""

COVID_FULL_GROUPED_DESCRIPTION = """Global COVID-19 data grouped by country/region and date. Contains daily confirmed cases, 
deaths, recovered cases, active cases, and new cases/deaths/recovered for each country/region. Includes WHO Region 
classification. Data spans from January 2020 onwards with country-level aggregation."""

DATASETS: Dict[str, Dict[str, Any]] = {
    "jpm_patient_data": {
        "description": JPM_PATIENT_DATA_DESCRIPTION,
        "code_name": "df_jpm_patients",  # name to bind in exec environment
        "storage": {
            "kind": "local_csv",
            "path": PROJECT_ROOT / "data" / "jpm_patient_data.csv",
            "read_params": {},
        },
        # Keep path for backward compatibility during migration
        "path": PROJECT_ROOT / "data" / "jpm_patient_data.csv",
    },
    "jamdas_patient_data": {
        "description": JAMDAS_PATIENT_DATA_DESCRIPTION,
        "code_name": "df_jamdas_patients",  # name to bind in exec environment
        "storage": {
            "kind": "local_csv",
            "path": PROJECT_ROOT / "data" / "jamdas_patient_data.csv",
            "read_params": {},
        },
        # Keep path for backward compatibility during migration
        "path": PROJECT_ROOT / "data" / "jamdas_patient_data.csv",
    },
    "covid_new_cases_daily": {
        "description": COVID_NEW_CASES_DAILY_DESCRIPTION,
        "code_name": "df_covid_daily",
        "storage": {
            "kind": "local_csv",
            "path": PROJECT_ROOT / "data" / "newly_confirmed_cases_daily.csv",
            "read_params": {},
        },
        # Keep path for backward compatibility during migration
        "path": PROJECT_ROOT / "data" / "newly_confirmed_cases_daily.csv",
    },
    "mr_activity_data": {
        "description": MR_ACTIVITY_DATA_DESCRIPTION,
        "code_name": "df_mr_activity",  # name to bind in exec environment
        "storage": {
            "kind": "local_csv",
            "path": PROJECT_ROOT / "data" / "mr_activity_data.csv",
            "read_params": {},
        },
        # Keep path for backward compatibility during migration
        "path": PROJECT_ROOT / "data" / "mr_activity_data.csv",
    },
    "covid_full_grouped": {
        "description": COVID_FULL_GROUPED_DESCRIPTION,
        "code_name": "df_covid_full_grouped",
        "storage": {
            "kind": "s3_csv",
            "bucket": AWS_S3_BUCKET,
            "key": "full_grouped.csv",
            "region": AWS_DEFAULT_REGION,
            "read_params": {},
        },
        "tags": ["covid", "global", "s3"],
    },
}
