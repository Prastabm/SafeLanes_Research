import pandas as pd
from datetime import timedelta

# Keywords to identify outdoor vs indoor locations for cities other than Baltimore
OUTDOOR_KEYWORDS = [
    'STREET', 'ROAD', 'ALLE', 'DRIVE', 'AVENUE', 'LANE', 'HIGHWAY', 'PARK', 'PARKWAY',
    'SIDEWALK', 'ALLEY', 'PAVEMENT', 'PLAZA', 'BRIDGE', 'OVERPASS', 'COURT', 'CIRCLE', 'BOULEVARD','STREET', 'SIDEWALK', 'DRIVEWAY', 'PARKING LOT', 'BUS STOP',
    'ROAD', 'FREEWAY', 'ALLEY', 'INTERSECTION', 'BRIDGE', 'HIGHWAY',
    'VEHICLE', 'OPEN AREA', 'OUTSIDE'
]
INDOOR_KEYWORDS = [
    'APARTMENT', 'RESIDENCE', 'HOUSE', 'HOME', 'STORE', 'SHOP', 'RESTAURANT', 'BAR',
    'HOTEL', 'HOSPITAL', 'SCHOOL', 'CLUB', 'OFFICE', 'BUILDING', 'GARAGE', 'STATION', 'CHURCH'
]

# Category normalization based on crime type or description keywords
def normalize_crime_category(raw_type: str, description: str) -> str:
    text = str(raw_type).upper() if pd.notnull(raw_type) else ''
    desc = str(description).upper() if pd.notnull(description) else ''
    combined = f"{text} {desc}"

    if 'HOMICIDE' in combined or 'MURDER' in combined:
        return 'Homicide'
    if 'ASSAULT' in combined:
        return 'Assault'
    if 'ROBBERY' in combined or 'THEFT' in combined or 'LARCENY' in combined:
        return 'Theft'
    if 'BURGLARY' in combined:
        return 'Burglary'
    if 'VEHICLE' in combined or 'MOTOR' in combined or 'CARJACK' in combined:
        return 'Vehicle Crime'
    if 'DRUG' in combined or 'NARCOTIC' in combined:
        return 'Drug Offense'
    if 'SEX' in combined or 'RAPE' in combined:
        return 'Sexual Offense'
    if 'FRAUD' in combined or 'FORGERY' in combined or 'EMBEZZ' in combined:
        return 'Fraud'
    # Default fallback category
    return 'Other'


def is_outdoor(location_desc: str) -> bool:
    """
    Return True if the location description indicates an outdoor setting.
    """
    if pd.isnull(location_desc):
        return False
    text = location_desc.upper()
    if any(ind in text for ind in INDOOR_KEYWORDS):
        return False
    return any(out in text for out in OUTDOOR_KEYWORDS)


# def load_baltimore(file_path: str = "datasets/baltimore.csv") -> pd.DataFrame:
#     df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
#     df = df[['CrimeDate', 'CrimeTime', 'Location 1', 'Description', 'Location', 'CrimeCode']].copy()
#     df[['Latitude', 'Longitude']] = df['Location 1']\
#         .str.extract(r"\(([^,]+), ([^)]+)\)")\
#         .astype(float)
#     df['datetime'] = pd.to_datetime(df['CrimeDate'] + ' ' + df['CrimeTime'], errors='coerce')
#     df.rename(columns={
#         'Description': 'description',
#         'Location': 'location_desc',
#         'CrimeCode': 'crime_type'
#     }, inplace=True)
#     codes = df['description'].str.split('/').str[-1].str.strip().str.upper()
#     df = df[codes == 'O']
#     # Normalize crime category
#     df['crime_category'] = df.apply(
#         lambda row: normalize_crime_category(row['crime_type'], row['description']), axis=1
#     )
#     df['city'] = 'Baltimore'
#     return df[['datetime', 'Latitude', 'Longitude', 'city', 'crime_category', 'description']].dropna()

def load_baltimore(file_path: str = "datasets/baltimore.csv",
                   code_map_path: str = "datasets/CRIME_CODES.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    df = df[['CrimeDate', 'CrimeTime', 'Location 1', 'Description', 'Location', 'CrimeCode']].copy()

    # Extract latitude and longitude
    df[['Latitude', 'Longitude']] = df['Location 1'] \
        .str.extract(r"\(([^,]+), ([^)]+)\)") \
        .astype(float)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['CrimeDate'] + ' ' + df['CrimeTime'], errors='coerce')

    # Rename columns for consistency
    df.rename(columns={
        'Description': 'description',
        'Location': 'location_desc',
        'CrimeCode': 'CrimeCode'
    }, inplace=True)

    # Filter by outdoor code in description (like "ROBBERY/O")
    codes = df['description'].str.split('/').str[-1].str.strip().str.upper()
    df = df[codes == 'O']

    # Load crime code mappings
    code_map = pd.read_csv(code_map_path)
    code_map.rename(columns={'CODE': 'CrimeCode', 'NAME_COMBINE': 'crime_code_desc'}, inplace=True)

    # Make sure both are strings for matching
    df['CrimeCode'] = df['CrimeCode'].astype(str).str.strip()
    code_map['CrimeCode'] = code_map['CrimeCode'].astype(str).str.strip()

    # Merge to get full crime code description
    df = df.merge(code_map[['CrimeCode', 'crime_code_desc']], how='left', on='CrimeCode')

    # Normalize crime category
    df['crime_category'] = df.apply(
        lambda row: normalize_crime_category(row['crime_code_desc'], row['description']), axis=1
    )

    df['city'] = 'Baltimore'
    return df[['datetime', 'Latitude', 'Longitude', 'city', 'crime_category', 'description']].dropna()


def load_boston(file_path: str = "datasets/boston.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    df = df[['OCCURRED_ON_DATE', 'HOUR', 'Lat', 'Long', 'OFFENSE_CODE_GROUP', 'OFFENSE_DESCRIPTION', 'Location']].copy()
    df['datetime'] = pd.to_datetime(df['OCCURRED_ON_DATE'], errors='coerce') + pd.to_timedelta(df['HOUR'], unit='h')
    df.rename(columns={
        'Lat': 'Latitude',
        'Long': 'Longitude',
        'OFFENSE_CODE_GROUP': 'crime_type',
        'OFFENSE_DESCRIPTION': 'description',
        'Location': 'location_desc'
    }, inplace=True)
    #df = df[df['location_desc'].apply(is_outdoor)]
    df['crime_category'] = df.apply(
        lambda row: normalize_crime_category(row['crime_type'], row['description']), axis=1
    )
    df['city'] = 'Boston'
    return df[['datetime', 'Latitude', 'Longitude', 'city', 'crime_category', 'description']].dropna()


def load_los_angeles(file_path: str = "datasets/los_angeles.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    df = df[['DATE OCC', 'TIME OCC', 'LAT', 'LON', 'Crm Cd Desc', 'Premis Desc', 'Part 1-2']].copy()
    df['date'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y', errors='coerce')
    time_str = df['TIME OCC'].astype(str).str.zfill(4)
    df['hours'] = time_str.str[:2].astype(int)
    df['minutes'] = time_str.str[2:].astype(int)
    df['datetime'] = (df['date'] + pd.to_timedelta(df['hours'], unit='h')
                      + pd.to_timedelta(df['minutes'], unit='m'))
    df.rename(columns={
        'LAT': 'Latitude',
        'LON': 'Longitude',
        'Crm Cd Desc': 'description',
        'Premis Desc': 'location_desc',
        'Part 1-2': 'crime_type'
    }, inplace=True)

    print("Before outdoor filter:", len(df))
    df = df[df['location_desc'].apply(is_outdoor)]
    print("After outdoor filter:", len(df))
    #df = df[df['location_desc'].apply(is_outdoor)]
    df['crime_category'] = df.apply(
        lambda row: normalize_crime_category(row['crime_type'], row['description']), axis=1
    )
    df['city'] = 'Los Angeles'
    return df[['datetime', 'Latitude', 'Longitude', 'city', 'crime_category', 'description']].dropna()


def load_new_york(file_path: str = "datasets/new_york.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
    df = df[[
        'CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'Latitude', 'Longitude',
        'LAW_CAT_CD', 'OFNS_DESC', 'PREM_TYP_DESC'
    ]].copy()
    df['datetime'] = pd.to_datetime(df['CMPLNT_FR_DT'] + ' ' + df['CMPLNT_FR_TM'], errors='coerce')
    df.rename(columns={
        'LAW_CAT_CD': 'crime_type',
        'OFNS_DESC': 'description',
        'PREM_TYP_DESC': 'location_desc'
    }, inplace=True)
    df = df[df['location_desc'].apply(is_outdoor)]
    df['crime_category'] = df.apply(
        lambda row: normalize_crime_category(row['crime_type'], row['description']), axis=1
    )
    df['city'] = 'New York'
    return df[['datetime', 'Latitude', 'Longitude', 'city', 'crime_category', 'description']].dropna()


def combine_datasets() -> pd.DataFrame:
    baltimore_df = load_baltimore()
    boston_df = load_boston()
    la_df = load_los_angeles()
    ny_df = load_new_york()

    combined_df = pd.concat([baltimore_df, boston_df, la_df, ny_df], ignore_index=True)
    combined_df.dropna(subset=['Latitude', 'Longitude', 'datetime', 'crime_category', 'description'], inplace=True)
    return combined_df


def save_cleaned_data():
    combined_df = combine_datasets()
    combined_df.to_csv("datasets/cleaned_crime_data3.csv", index=False)
    print("✅ Cleaned and combined data saved to data/cleaned_crime_data2.csv")
