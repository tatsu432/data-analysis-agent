"""Dataset registry for MCP data analysis tools."""

from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent

JAMDAS_PATIENT_DATA_DESCRIPTION = """The patient data for each product with information about the at risk patients 
and ddi prescriptions. This dataset only covers GP (the medical institution whose number of beds is lower than 20).
 It does not have the information about each prefecture's patinet but overall patient."""

JPM_PATIENT_DATA_DESCRIPTION = """The patient data for each product with information about GP (the medical 
institution whose number of beds is lower than 20) or HP (the medical institution whose number of beds is higher than 20).
It does not have the information about each prefecture's patinet but overall patient."""

COVID_NEW_CASES_DAILY_DESCRIPTION = (
    "COVID-19 newly confirmed cases daily data for Japanese prefectures (local CSV)."
)

DATASETS: Dict[str, Dict[str, Any]] = {
    "jpm_patient_data": {
        "path": PROJECT_ROOT / "data" / "jpm_patient_data.csv",
        "description": JPM_PATIENT_DATA_DESCRIPTION,
        "code_name": "df_jpm_patients",  # name to bind in exec environment
    },
    "jamdas_patient_data": {
        "path": PROJECT_ROOT / "data" / "jamdas_patient_data.csv",
        "description": JAMDAS_PATIENT_DATA_DESCRIPTION,
        "code_name": "df_jamdas_patients",  # name to bind in exec environment
    },
    "covid_new_cases_daily": {
        "path": PROJECT_ROOT / "data" / "newly_confirmed_cases_daily.csv",
        "description": COVID_NEW_CASES_DAILY_DESCRIPTION,
        "code_name": "df_covid_daily",
    },
}
