
import pandas as pd
file_name='Urban_Parents_Club_Data.xlsx'
Sheet_name_n='Nanny Details'
Sheet_name_p='Parent Details'
df_nanny=pd.read_excel(file_name,Sheet_name_n)
df_parent=pd.read_excel(file_name,Sheet_name_p)
print("Checking the nanny data types")
print(df_nanny.dtypes)
print('------------------------------------------------')
print("Checking the parent data types")
print(df_parent.dtypes)

# Count missing values (%) for EACH column

missing_percent_n= (df_nanny.isnull().sum() /df_nanny.shape[0]) *100
print(f"The paercent of missing value per Column in nanny is: ")
print(missing_percent_n[missing_percent_n >0].round(2))

print('-----------------------------------------------------')

missing_percent_p= (df_parent.isnull().sum() /df_parent.shape[0]) *100
print(f"The paercent of missing value per Column in parent is: ")
print(missing_percent_p[missing_percent_p >0].round(2))








missing_percent_n= (df_nanny.isnull().sum() /df_nanny.shape[0]) *100
print(f"The paercent of missing value per Column in nanny is: ")
print(missing_percent_n[missing_percent_n >0].round(2))

print('-----------------------------------------------------')

missing_percent_p= (df_parent.isnull().sum() /df_parent.shape[0]) *100
print(f"The paercent of missing value per Column in parent is: ")
print(missing_percent_p[missing_percent_p >0].round(2))









import pandas as pd
import numpy as np
from datetime import date

# Set a fixed date for age calculations
TODAY = date(2025, 10, 17)

print("--- DATA VALIDATION & INCONSISTENCY REPORT ---")

# --- 1. SETUP: AGE CALCULATION & TYPE CONVERSIONS ---
# (Assumes df_nanny is already loaded)
# ----------------------------------------------------------------------------------

# Convert Date of Birth to datetime objects
df_nanny['Date of Birth'] = pd.to_datetime(df_nanny['Date of Birth'], errors='coerce')

# Custom function to calculate age
def calculate_age(born):
    if pd.isna(born):
        return np.nan
    age = TODAY.year - born.year
    if (TODAY.month, TODAY.day) < (born.month, born.day):
        return age - 1
    return age

df_nanny['Age'] = df_nanny['Date of Birth'].apply(calculate_age)

# Convert key numeric columns to numeric, handling errors
df_nanny['Experience (Years)'] = pd.to_numeric(df_nanny['Experience (Years)'], errors='coerce')
df_nanny['Working Hours'] = pd.to_numeric(df_nanny['Working Hours'], errors='coerce')


# --- 2. CHILDREN AGE VALIDATION & MINIMUM AGE PROOF ---
# ----------------------------------------------------------------------------------
CHILDREN_AGE_COL = 'Children Age(s)'

# Standardize the format (pipe separation)
df_nanny[CHILDREN_AGE_COL] = (
    df_nanny[CHILDREN_AGE_COL].astype(str)
    .str.replace(r'[,\s]+', '|', regex=True)
    .str.strip('|')
)

# Inconsistency Check: Children Age Range (typos > 20)
LOGICAL_MAX_CHILD_AGE = 20
children_age_inconsistent = df_nanny[
    df_nanny[CHILDREN_AGE_COL].apply(
        lambda x: any(
            pd.to_numeric(age, errors='coerce') > LOGICAL_MAX_CHILD_AGE
            for age in str(x).split('|') if age and str(x).lower() != 'nan'
        )
    )
].copy()

# PROOF: Calculate and display the minimum age found in the Children Age column
if CHILDREN_AGE_COL in df_nanny.columns:
    # 1. Expand the pipe-separated column into individual cells
    all_child_ages = df_nanny[CHILDREN_AGE_COL].str.split('|', expand=True).stack()
    # 2. Convert all ages to numeric and find the minimum
    min_age_found = pd.to_numeric(all_child_ages, errors='coerce').min()

    # Store the proof string
    min_age_proof = f"Minimum Age Found in Children Data: {int(min_age_found) if pd.notna(min_age_found) else 'N/A'}"
else:
    min_age_proof = "Minimum Age Found in Children Data: N/A (Column Missing)"


# --- 3. BUSINESS RULE INCONSISTENCY CHECKS ---
# ----------------------------------------------------------------------------------

# Rule A: Nanny Age (18-70 years)
age_inconsistent = df_nanny[
    df_nanny['Age'].notna() & (
        (df_nanny['Age'] < 18) | (df_nanny['Age'] > 70)
    )
].copy()

# Rule B1: Experience (0-50 years)
experience_range_inconsistent = df_nanny[
    df_nanny['Experience (Years)'].notna() &
     (
    (df_nanny['Experience (Years)'] < 0) | (df_nanny['Experience (Years)'] > 50)
    )
].copy()

# Rule B2: Experience must be less than Age
experience_age_compare = df_nanny[
    df_nanny['Experience (Years)'].notna() & df_nanny['Age'].notna() & (
    df_nanny['Experience (Years)'] >= df_nanny['Age'])
    ].copy()

# Rule D: Working Hours (1-24)
hours_incorrect = df_nanny[
    df_nanny['Working Hours'].notna() & (
    (df_nanny['Working Hours'] < 1) | (df_nanny['Working Hours'] > 24)
    )
].copy()


# --- FEATURE VALIDITY CHECKS ---
# ----------------------------------------------------------------------------------

# Check 1: Languages look valid? (i.e., not just numbers or excessive symbols)
def check_invalid_text(series):
    # Find entries that contain excessive non-alphabetic/non-space/non-comma characters.
    return series.astype(str).str.contains('[^a-zA-Z\s,]').sum()

# We assume 'Languages' survived the automated drop
invalid_languages_count = 0
if 'Languages' in df_nanny.columns:
    # After cleaning, look for non-language text (e.g., symbols, codes)
    invalid_languages_count = check_invalid_text(df_nanny['Languages'])

# Check 2: Hours add up? (impossibly high?) - Already covered by Rule D (Working Hours)

# Check 3: Budget values make sense? - SKIPPED (No budget column)


# --- 4. DUPLICATE REMOVAL ---
# ----------------------------------------------------------------------------------

print("\n--- 4. REMOVING DUPLICATE RECORDS ---")

# --- Nanny Data Duplicates ---
nanny_cols_for_subset = ['Full Name', 'Date of Birth', 'Home Location']
# Filter the list to only include columns that exist after cleaning/dropping
nanny_subset = [col for col in nanny_cols_for_subset if col in df_nanny.columns]

rows_before_nanny = df_nanny.shape[0]
df_nanny.drop_duplicates(subset=nanny_subset, keep='first', inplace=True)
rows_after_nanny = df_nanny.shape[0]
nanny_duplicates_removed = rows_before_nanny - rows_after_nanny

print(f"Nanny Records removed: {nanny_duplicates_removed}. Final rows: {rows_after_nanny}")


# --- Parent Data Duplicates ---
parent_cols_for_subset = ['Invitee Name', 'Invitee First Name', 'Event Created Date & Time']
# Filter the list to only include columns that exist after cleaning/dropping
parent_subset = [col for col in parent_cols_for_subset if col in df_parent.columns]

rows_before_parent = df_parent.shape[0]
df_parent.drop_duplicates(subset=parent_subset, keep='first', inplace=True)
rows_after_parent = df_parent.shape[0]
parent_duplicates_removed = rows_before_parent - rows_after_parent

print(f"Parent Records removed: {parent_duplicates_removed}. Final rows: {rows_after_parent}")


# --- 5. FINAL REPORTING ---
# ----------------------------------------------------------------------------------
print("\n--- FINAL INCONSISTENCY COUNTS ---")

# Display the proof line first
print(f"\n[DATA PROOF] {min_age_proof}\n")

report_series = pd.Series({
    'Nanny Age (<18 or >70)': age_inconsistent.shape[0],
    'Experience Range (<0 or >50)': experience_range_inconsistent.shape[0],
    'Experience >= Age': experience_age_compare.shape[0],
    'Children Age (Typos >20)': children_age_inconsistent.shape[0],
    'Working Hours (<1 or >24)': hours_incorrect.shape[0],
    'Language Validity (Invalid Chars)': invalid_languages_count
})

print(report_series)














# import pandas as pd
import numpy as np
from datetime import date
# pip install fuzzywuzzy[speedup]
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# --- SETUP: FILE READING ---
file_name='Urban_Parents_Club_Data.xlsx'
Sheet_name_n='Nanny Details'
Sheet_name_p='Parent Details'
df_nanny=pd.read_excel(file_name,Sheet_name_n)
df_parent=pd.read_excel(file_name,Sheet_name_p)

# Set a fixed date for age calculations
TODAY = date(2025, 10, 17)

print("--- STARTING COMPLETE DATA CLEANING PIPELINE ---")

# --- CORE FUNCTIONS ---

def calculate_age(born):
    if pd.isna(born):
        return np.nan
    age = TODAY.year - born.year
    if (TODAY.month, TODAY.day) < (born.month, born.day):
        return age - 1
    return age

TEMP_FILL = 'MISSING_LANG'
def standardize_language_column(df, column_name, mapping):
    if column_name not in df.columns:
        return df

    # 1. Fill NaN with temporary string, convert to string, strip, and lowercase
    df[column_name] = df[column_name].fillna(TEMP_FILL).astype(str).str.strip().str.lower()

    # 2. Apply the language standardization map
    df[column_name] = df[column_name].replace(mapping, regex=True)

    # 3. Convert back to Title Case and replace the temporary string with NaN
    df[column_name] = df[column_name].str.title()
    df[column_name] = df[column_name].replace(TEMP_FILL.title(), np.nan)
    return df

def automate_typo_error(df, column_name, master_list, threshold=90):
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found. Skipping typo correction.")
        return df

    # Prepare data for fuzzy matching: strip, lowercase, handle NaN
    series = df[column_name].astype(str).str.strip().str.lower()
    unique_values = series.dropna().unique()

    correction_map = {}

    # Build the correction map
    for entry in unique_values:
        # Skip the 'nan' string value which is the placeholder for NaN
        if entry == 'nan':
            continue

        match, score = process.extractOne(entry, master_list)

        # Apply correction if score meets threshold
        if score >= threshold:
            correction_map[entry] = match.title()
        else:
            # Keep original entry if score is too low
            correction_map[entry] = entry.title()

    # Apply map to the column
    # Ensure all original values (including NaNs) are mapped
    corrected_series = series.str.lower().map(correction_map)

    # Map back any NaN values that might have been lost in the map
    df[column_name] = corrected_series
    df[column_name] = df[column_name].replace(TEMP_FILL.title(), np.nan)

    print(f"Automated Cleanup Complete for {column_name}")
    return df

def automated_column_dropper(df, df_name):
    total_rows = df.shape[0]
    MIN_NON_NULL_PERCENT = 0.25
    required_non_null = int(total_rows * MIN_NON_NULL_PERCENT)

    df_cleaned = df.dropna(axis=1, thresh=required_non_null)

    dropped_count = df.shape[1] - df_cleaned.shape[1]
    print(f"[{df_name}] Total Rows: {total_rows}. Required Non nulls: {required_non_null}")
    print(f"Dropped {dropped_count} columns that were less than {MIN_NON_NULL_PERCENT*100}% full")
    return df_cleaned


# --- 1. DETAILED CLEANUP (Before Dropping Columns) ---

# A. Age Calculation & Conversions
df_nanny['Date of Birth'] = pd.to_datetime(df_nanny['Date of Birth'], errors='coerce')
df_nanny['Age'] = df_nanny['Date of Birth'].apply(calculate_age)
df_nanny['Experience (Years)'] = pd.to_numeric(df_nanny['Experience (Years)'], errors='coerce')
df_nanny['Working Hours'] = pd.to_numeric(df_nanny['Working Hours'], errors='coerce')
df_parent["Baby's Age"] = pd.to_numeric(df_parent["Baby's Age"], errors='coerce')
print("✅ 1. Calculated Age and converted numeric columns.")

# B. Language Standardization
stardadization_map = {"bengaluru": "Bangalore", "portugese": "Portuguese", "spanih": "Spanish"}
Language_nanny = ['Languages']
Language_parent = ['Response 4'] # Assuming Response 4 contains location or language
for cols in Language_nanny:
    df_nanny = standardize_language_column(df_nanny, cols, stardadization_map)
for cols in Language_parent:
    df_parent = standardize_language_column(df_parent, cols, stardadization_map)
print("✅ 2. Standardized language/location casing.")

# C. Fuzzy Matching (Automated Typos)
NANNY_MASTER_LISTS = {
    'Education Level': ['High School', 'Bachelor', 'Masters', 'Diploma', 'PhD', 'None'],
    'Marital Status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'Type of Role': ['Full-time', 'Part-time', 'Live-in', 'Nanny', 'School', 'Live-out'],
    'Aadhaar Collected?': ['Yes', 'No'],
    'Police Verified?': ['Yes', 'No'],
    'Health Certificate Uploaded?': ['Yes', 'No'],
    'Home Location': ['Vivek Nagar', 'Bommanahalli', 'Rajajinagar', 'Koramangala', 'Mysore', 'Jharkhand'] # Use sample locations
}
PARENT_MASTER_LISTS = {
    'Type of Nanny': ['Live-in', 'Live-out', 'Full-time', 'Part-time'],
    'Parent Location': ['Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad']
}

for col, master_list in NANNY_MASTER_LISTS.items():
    df_nanny = automate_typo_error(df_nanny, col, master_list, threshold=90)
for col, master_list in PARENT_MASTER_LISTS.items():
    df_parent = automate_typo_error(df_parent, col, master_list, threshold=90)
print("✅ 3. Applied automated fuzzy typo correction.")

# D. Flagging (Retaining the original DataFrames for now)
Date_cols_to_flag = ['Last Contact Date', 'Next Follow-up Date', 'Last Updated', 'Created Date']
for col in Date_cols_to_flag:
    if col in df_nanny.columns:
        flag_name = f'flag_is_{col.replace(" ", "_").replace("–", "")}_missing'
        df_nanny[flag_name] = df_nanny[col].isnull().astype(int)
print("✅ 4. Flagged key date columns.")


# --- 2. AUTOMATED COLUMN REDUCTION ---
df_nanny = automated_column_dropper(df_nanny, 'Nanny')
df_parent = automated_column_dropper(df_parent, 'Parent')


# --- 3. FINAL IMPUTATION (Only on SURVIVING columns) ---
# Nanny Imputation
if 'Experience (Years)' in df_nanny.columns:
    df_nanny['Experience (Years)'] = df_nanny['Experience (Years)'].fillna(0)
if 'Working Hours' in df_nanny.columns:
    df_nanny['Working Hours'] = df_nanny['Working Hours'].fillna(0)
if 'No of Children' in df_nanny.columns:
    df_nanny['No of Children'] = df_nanny['No of Children'].fillna(0)
if 'Children Age(s)' in df_nanny.columns:
    df_nanny['Children Age(s)'] = df_nanny['Children Age(s)'].fillna('0')
if 'Languages' in df_nanny.columns:
    df_nanny['Languages'] = df_nanny['Languages'].fillna('Unknown')

# Parent Imputation
PARENT_CAT_COLS = ['Type of Nanny', "Baby's Age", 'Parent Location']
for col in PARENT_CAT_COLS:
    if col in df_parent.columns:
        df_parent[col] = df_parent[col].fillna('Unknown')
print("✅ 5. Final imputation of missing numerical/categorical values.")


# --- 4. DUPLICATE REMOVAL ---
# Define the required key identifier columns for Nanny data
REQUIRED_NANNY_IDENTIFIERS = ['Full Name', 'Date of Birth', 'Home Location']
nanny_subset = [col for col in REQUIRED_NANNY_IDENTIFIERS if col in df_nanny.columns]

# Define the required key identifier columns for Parent data
REQUIRED_PARENT_IDENTIFIERS = ['Invitee Name', 'Event Created Date & Time', 'Parent Location']
parent_subset = [col for col in REQUIRED_PARENT_IDENTIFIERS if col in df_parent.columns]

# Check if essential subsets are available before dropping
if not nanny_subset:
    print("❌ ERROR: Nanny Data has no primary key columns left for duplicate check.")
elif not parent_subset:
    print("❌ ERROR: Parent Data has no primary key columns left for duplicate check.")
else:
    rows_before_nanny = df_nanny.shape[0]
    df_nanny.drop_duplicates(subset=nanny_subset, keep='first', inplace=True)
    rows_after_nanny = df_nanny.shape[0]
    print(f"✅ 6. Duplicates removed from Nanny data. Rows removed: {rows_before_nanny - rows_after_nanny}")

    rows_before_parent = df_parent.shape[0]
    df_parent.drop_duplicates(subset=parent_subset, keep='first', inplace=True)
    rows_after_parent = df_parent.shape[0]
    print(f"✅ 7. Duplicates removed from Parent data. Rows removed: {rows_before_parent - rows_after_parent}")


# --- 5. FINAL EXPORT ---
df_nanny.to_csv('nanny_details_cleaned.csv', index=False)
df_parent.to_csv('parent_details_cleaned.csv', index=False)
print("\n--- PIPELINE COMPLETE! ---")
print("Saved files: nanny_details_cleaned.csv and parent_details_cleaned.csv")












file_name='Urban_Parents_Club_Data.xlsx'
Sheet_name_n='Nanny Details'
Sheet_name_p='Parent Details'
df_nanny=pd.read_excel(file_name,Sheet_name_n)
df_parent=pd.read_excel(file_name,Sheet_name_p)

import pandas as pd


print("\n## Distribution of Parent Locations (Top Request Hotspots)")
print("-------------------------------------------------------")
print(df_parent['Parent Location'].value_counts(dropna=False).head(5))



# Create distribution of baby AGES (typical age range?)
print("\n## Distribution of Baby Ages (Typical Age Range)")
print("---------------------------------------------")

Baby_AGE="Baby\'s Age"
df_parent[Baby_AGE] = pd.to_numeric(df_parent[Baby_AGE], errors='coerce')
print(df_parent[Baby_AGE].describe())

# Create distribution of nanny AVAILABILITY (full-time vs part-time?)
print("\n## Distribution of Nanny Preferred Working Hours")
print("-----------------------------------------------")
# Describes the central tendency and spread of preferred working hours.
if 'Working Hours' in df_nanny.columns:
    print(df_nanny['Working Hours'].describe())
else:
    print("Warning: Column 'Working Hours' not found.")

# Create distribution of nanny LOCATIONS (geographic spread?)
print("\n## Distribution of Nanny Home Locations")
print("--------------------------------------")
if 'Home Location' in df_nanny.columns:
    print(df_nanny['Home Location'].value_counts(dropna=False).head(5))
else:
    print("Warning: Column 'Home Location' not found.")
print("\n## Distribution of Nanny HOURLY RATES (External Market Rate)")
print("---------------------------------------------------------")
print("Data not available internally. Using Bengaluru Market Benchmarks (2024/2025):")
print("\n--- HOURLY RATE DISTRIBUTION (INR/Hour) ---")
print("Low-End (Basic/Part-Time): 150 - 200")
print("Average Market Rate:       200 - 250 (ERI Average: 233)")
print("High-End (Specialized):    300 - 500+")
print("\n--- MONTHLY SALARY RANGE (Full-Time/Live-In) ---")
print("8-10 Hour Shifts:          12,000 - 18,000")
print("Live-In/24 Hour Care:      16,000 - 25,000+")


if 'Working Hours' in df_nanny.columns:
    print(df_nanny['Working Hours'].describe())
else:
    print("Warning: Column 'Working Hours' not found.")


















import pandas as pd
import io
import re # Import the regular expression module

# --- 1. WEIGHT DEFINITION (Copied from MVP) ---
WEIGHTS = {
    "Location_Match": 0.30,
    "Language_Match": 0.20,
    "Experience_Match": 0.20,
    "Availability_Match": 0.15,
    "Travel_Willingness": 0.15
}

# --- 2. SCORING LOGIC FUNCTIONS (Copied from MVP) ---

def score_location(parent_loc, nanny_pref_loc):
    """
    Component 1 (30%): Exact neighborhood match (100) or miss (0).
    """
    if str(parent_loc).strip().lower() == str(nanny_pref_loc).strip().lower():
        return 100
    return 0

def score_language(parent_lang, nanny_languages):
    """
    Component 2 (20%): If the Nanny speaks ANY of the Parent's required languages, it's a 100.
    """
    # Parent's language requirement is usually single (e.g., 'Tamil') or comma-separated.
    parent_langs = [l.strip().lower() for l in str(parent_lang).split(',')]
    # Nanny's language list might include proficiency (e.g., 'English (Basic)'). We strip the proficiency.
    nanny_langs = [l.split('(')[0].strip().lower() for l in str(nanny_languages).split(',')]

    for p_lang in parent_langs:
        if p_lang and p_lang in nanny_langs:
            return 100
    return 0

def score_experience(baby_age_str, nanny_exp_years):
    """
    Component 3 (20%): Scores Nanny experience against baby's age/needs.
    Rule: Requires minimal experience (0) for babies under 3 years, and 3+ years experience for older children.
    """
    try:
        # The 'np' (Not Provided) values will be NaN after loading, but we ensure robustness here.
        nanny_exp = float(nanny_exp_years)
    except (ValueError, TypeError):
        nanny_exp = 0.0

    baby_age_lower = str(baby_age_str).lower()

    # --- Robust Age Extraction Logic ---
    # Handle entries like "twin 1.5 years", "2 & 4 years", or just "6 months"

    # Extract all numbers (including decimals) from the age string
    numbers = re.findall(r'\d+\.?\d*', baby_age_lower)

    # Simple check for age based on keywords, defaulting to the first number found if any.
    if 'months' in baby_age_lower:
        # If age is in months, it is definitely < 3 years old.
        required_exp_years = 0
    elif numbers and 'years' in baby_age_lower:
        # If we found numbers and 'years', use the first number as the age for calculation
        try:
            baby_age_years = float(numbers[0])
            if baby_age_years < 3:
                required_exp_years = 0
            else:
                required_exp_years = 3
        except ValueError:
            # Fallback if number extraction fails unexpectedly
            required_exp_years = 3 # Assume older child if parsing is ambiguous
    else:
        # For ambiguous cases like 'np' or just 'twin', assume default requirement of 3+ years
        required_exp_years = 3
    # --- End Robust Age Extraction Logic ---

    # --- Scoring based on requirement ---

    if 'months' in baby_age_lower:
        # Check for babies 0-2 years (e.g., '6 months')
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 0:
        # Case: Age is < 3 years, minimal experience required
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 3:
        # Case: Age is >= 3 years (or complex age, defaulting to high need)
        if nanny_exp >= 3:
            return 100

    # Neutral score for complex cases, or where the rule is not a perfect match
    return 50

def score_availability(parent_type, nanny_hours_str):
    """
    Component 4 (15%): Scores based on type of nanny needed vs. hours available.
    """
    try:
        # Convert hours to float for comparison. 'np' values will fail gracefully to 0.
        nanny_hours_float = float(nanny_hours_str)
    except (ValueError, TypeError):
         nanny_hours_float = 0

    parent_type = str(parent_type).lower()

    if '24 hours' in parent_type:
        # 24hr live-in is a big ask. Assuming 20+ hours needed for a 100 score.
        if nanny_hours_float >= 20:
            return 100
        return 0 # Miss if not available for high hours

    elif 'full time' in parent_type or 'full time nanny' in parent_type:
        # Full time (6 to 10 hours) requires 10+ hours availability
        if nanny_hours_float >= 10:
            return 100
        return 40 # Partial miss if only available for less (e.g., 6 hours)

    elif 'part time' in parent_type or 'part time nanny' in parent_type:
        # Part time (0 to 6 hours) requires at least 6 hours
        if nanny_hours_float >= 6:
            return 100
        return 60 # Small miss if less than 6 hours

    return 20 # Low score for ambiguous types

def score_travel(travel_willingness):
    """
    Component 5 (15%): Scores based on Nanny's willingness to travel/commute.
    The column 'Willing to Travel (km)' in the data suggests a True/False value.
    """
    if str(travel_willingness).lower() == 'true':
        return 100
    return 50 # Not willing to travel is a constraint, hence 50

def score_single_match(parent, nanny):
    """
    Calculates the final weighted score for one Parent-Nanny pair (0-100).
    Returns: (final_score, component_scores_dict)
    """
    # 1. Calculate individual component scores (0-100)
    score_results = {
        "Location_Match": score_location(
            parent['Parent Location'],
            nanny['Parent Location'] # This column holds the Nanny's preferred work area
        ),
        "Language_Match": score_language(
            parent['Language'],
            nanny['Languages']
        ),
        "Experience_Match": score_experience(
            parent["Baby's Age"],
            nanny['Experience (Years)']
        ),
        "Availability_Match": score_availability(
            parent['Type of Nanny'],
            nanny['Working Hours']
        ),
        "Travel_Willingness": score_travel(
            nanny['Willing to Travel (km)']
        )
    }

    # 2. Apply weights and calculate final score
    final_score = 0
    for component, weight in WEIGHTS.items():
        component_score = score_results.get(component, 0)
        # Formula: Score * Weight
        weighted_contribution = component_score * weight
        final_score += weighted_contribution

    return round(final_score, 2), score_results


def clean_column_names(df):
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

# --- 3. EXECUTION BLOCK FOR COLAB ---

if __name__ == '__main__':
    print("Please upload the two EXCEL files when prompted:")

    # Define common robust read parameters
    # NOTE: read_excel is used instead of read_csv for robustness with Excel files.
    READ_EXCEL_PARAMS = {
        'sheet_name': 0, # Read the first sheet
        'header': 0,     # Header is in the first row
    }

    # Colab File Upload (Parent Data)
    print("\n--- Uploading Parent Data (Parent.data.xlsx - Sheet1.csv) ---")
    uploaded_parent=0
    if not uploaded_parent:
        print("Parent file upload failed. Exiting.")
    else:
        parent_file_name = list(uploaded_parent.keys())[0]
        # FIX: Use pd.read_excel with BytesIO for robust Excel reading
        parent_df = pd.read_excel('parent.data.xlsx')
        parent_df = clean_column_names(parent_df)

        # Colab File Upload (Nanny Data)
        print("\n--- Uploading Nanny Data (Nanny.data.xlsx - Sheet1.csv) ---")
        uploaded_nanny = files.upload()

        if not uploaded_nanny:
            print("Nanny file upload failed. Exiting.")
        else:
            nanny_file_name = list(uploaded_nanny.keys())[0]
            # FIX: Use pd.read_excel with BytesIO for robust Excel reading
            nanny_df = pd.read_excel('nanny.data.xlsx')
            nanny_df = clean_column_names(nanny_df)

            print(f"\nSuccessfully loaded {len(parent_df)} Parents and {len(nanny_df)} Nannies.")
            print("-" * 50)

            # --- Perform Full Matching ---

            # These columns are now guaranteed to be clean due to clean_column_names()
            PARENT_NAME_COL = 'Invitee Name'
            NANNY_NAME_COL = 'Full Name'

            match_results = []

            # Iterate through every parent and every nanny to find all possible scores
            for parent_index, parent_row in parent_df.iterrows():
                for nanny_index, nanny_row in nanny_df.iterrows():

                    # Calculate the match score using the MVP function
                    final_score, component_scores = score_single_match(parent_row, nanny_row)

                    # Store all relevant details
                    result = {
                        'Parent Name': parent_row[PARENT_NAME_COL],
                        'Nanny Name': nanny_row[NANNY_NAME_COL],
                        'Final Score': final_score,
                    }
                    result.update(component_scores) # Add the individual component scores

                    match_results.append(result)

            # Create the results DataFrame
            results_df = pd.DataFrame(match_results)

            # --- HOUR 3: TOP MATCHES PER PARENT ---

            NUM_PARENTS_TO_SHOW = 3
            TOP_K_MATCHES = 3

            print("\n\n--- HOUR 3: TOP MATCHES PER PARENT ---")

            # Iterate through the first three parents in the parent DataFrame
            for i in range(NUM_PARENTS_TO_SHOW):
                if i >= len(parent_df):
                    break # Stop if there are fewer than 3 parents

                # Extract the name and location for the current parent
                current_parent_name = parent_df.iloc[i][PARENT_NAME_COL]
                current_parent_loc = parent_df.iloc[i]['Parent Location']

                # Filter results for this parent and sort to get the top matches
                parent_specific_matches = results_df[results_df['Parent Name'] == current_parent_name].sort_values(
                    by='Final Score',
                    ascending=False
                ).head(TOP_K_MATCHES)

                # Print the parent header line
                print(f"\nParent {i+1} ({current_parent_loc}): {current_parent_name}")

                # Print the top N matches in the requested format
                for _, match_row in parent_specific_matches.iterrows():
                    nanny_name = match_row['Nanny Name']
                    # Cast the score to an integer as shown in the example output
                    score = int(match_row['Final Score'])
                    print(f"  {nanny_name} ({score})")

            print("-----------------------------------------")

            # --- Display Top Matches (The original Hour 2 output, now removed as Hour 3 is the focus) ---
            # You can still access the results_df in a new cell if needed.
            print("\nNote: The full 'results_df' DataFrame is available for further analysis in a new cell.")

















import pandas as pd
import io
import re # Import the regular expression module

# --- 1. WEIGHT DEFINITION (Copied from MVP) ---
WEIGHTS = {
    "Location_Match": 0.30,
    "Language_Match": 0.20,
    "Experience_Match": 0.20,
    "Availability_Match": 0.15,
    "Travel_Willingness": 0.15
}

# --- 2. SCORING LOGIC FUNCTIONS (Copied from MVP) ---

def score_location(parent_loc, nanny_pref_loc):
    """
    Component 1 (30%): Exact neighborhood match (100) or miss (0).
    """
    if str(parent_loc).strip().lower() == str(nanny_pref_loc).strip().lower():
        return 100
    return 0

def score_language(parent_lang, nanny_languages):
    """
    Component 2 (20%): If the Nanny speaks ANY of the Parent's required languages, it's a 100.
    """
    # Parent's language requirement is usually single (e.g., 'Tamil') or comma-separated.
    parent_langs = [l.strip().lower() for l in str(parent_lang).split(',')]
    # Nanny's language list might include proficiency (e.g., 'English (Basic)'). We strip the proficiency.
    nanny_langs = [l.split('(')[0].strip().lower() for l in str(nanny_languages).split(',')]

    for p_lang in parent_langs:
        if p_lang and p_lang in nanny_langs:
            return 100
    return 0

def score_experience(baby_age_str, nanny_exp_years):
    """
    Component 3 (20%): Scores Nanny experience against baby's age/needs.
    Rule: Requires minimal experience (0) for babies under 3 years, and 3+ years experience for older children.
    """
    try:
        # The 'np' (Not Provided) values will be NaN after loading, but we ensure robustness here.
        nanny_exp = float(nanny_exp_years)
    except (ValueError, TypeError):
        nanny_exp = 0.0

    baby_age_lower = str(baby_age_str).lower()

    # --- Robust Age Extraction Logic ---
    # Handle entries like "twin 1.5 years", "2 & 4 years", or just "6 months"

    # Extract all numbers (including decimals) from the age string
    numbers = re.findall(r'\d+\.?\d*', baby_age_lower)

    # Simple check for age based on keywords, defaulting to the first number found if any.
    if 'months' in baby_age_lower:
        # If age is in months, it is definitely < 3 years old.
        required_exp_years = 0
    elif numbers and 'years' in baby_age_lower:
        # If we found numbers and 'years', use the first number as the age for calculation
        try:
            baby_age_years = float(numbers[0])
            if baby_age_years < 3:
                required_exp_years = 0
            else:
                required_exp_years = 3
        except ValueError:
            # Fallback if number extraction fails unexpectedly
            required_exp_years = 3 # Assume older child if parsing is ambiguous
    else:
        # For ambiguous cases like 'np' or just 'twin', assume default requirement of 3+ years
        required_exp_years = 3
    # --- End Robust Age Extraction Logic ---

    # --- Scoring based on requirement ---

    if 'months' in baby_age_lower:
        # Check for babies 0-2 years (e.g., '6 months')
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 0:
        # Case: Age is < 3 years, minimal experience required
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 3:
        # Case: Age is >= 3 years (or complex age, defaulting to high need)
        if nanny_exp >= 3:
            return 100

    # Neutral score for complex cases, or where the rule is not a perfect match
    return 50

def score_availability(parent_type, nanny_hours_str):
    """
    Component 4 (15%): Scores based on type of nanny needed vs. hours available.
    """
    try:
        # Convert hours to float for comparison. 'np' values will fail gracefully to 0.
        nanny_hours_float = float(nanny_hours_str)
    except (ValueError, TypeError):
         nanny_hours_float = 0

    parent_type = str(parent_type).lower()

    if '24 hours' in parent_type:
        # 24hr live-in is a big ask. Assuming 20+ hours needed for a 100 score.
        if nanny_hours_float >= 20:
            return 100
        return 0 # Miss if not available for high hours

    elif 'full time' in parent_type or 'full time nanny' in parent_type:
        # Full time (6 to 10 hours) requires 10+ hours availability
        if nanny_hours_float >= 10:
            return 100
        return 40 # Partial miss if only available for less (e.g., 6 hours)

    elif 'part time' in parent_type or 'part time nanny' in parent_type:
        # Part time (0 to 6 hours) requires at least 6 hours
        if nanny_hours_float >= 6:
            return 100
        return 60 # Small miss if less than 6 hours

    return 20 # Low score for ambiguous types

def score_travel(travel_willingness):
    """
    Component 5 (15%): Scores based on Nanny's willingness to travel/commute.
    The column 'Willing to Travel (km)' in the data suggests a True/False value.
    """
    if str(travel_willingness).lower() == 'true':
        return 100
    return 50 # Not willing to travel is a constraint, hence 50

def score_single_match(parent, nanny):
    """
    Calculates the final weighted score for one Parent-Nanny pair (0-100).
    Returns: (final_score, component_scores_dict)
    """
    # 1. Calculate individual component scores (0-100)
    score_results = {
        "Location_Match": score_location(
            parent['Parent Location'],
            nanny['Parent Location'] # This column holds the Nanny's preferred work area
        ),
        "Language_Match": score_language(
            parent['Language'],
            nanny['Languages']
        ),
        "Experience_Match": score_experience(
            parent["Baby's Age"],
            nanny['Experience (Years)']
        ),
        "Availability_Match": score_availability(
            parent['Type of Nanny'],
            nanny['Working Hours']
        ),
        "Travel_Willingness": score_travel(
            nanny['Willing to Travel (km)']
        )
    }

    # 2. Apply weights and calculate final score
    final_score = 0
    for component, weight in WEIGHTS.items():
        component_score = score_results.get(component, 0)
        # Formula: Score * Weight
        weighted_contribution = component_score * weight
        final_score += weighted_contribution

    return round(final_score, 2), score_results


def clean_column_names(df):
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

# --- 3. EXECUTION BLOCK FOR COLAB ---

if __name__ == '__main__':
    print("Please upload the two EXCEL files when prompted:")

    # Define common robust read parameters
    # NOTE: read_excel is used instead of read_csv for robustness with Excel files.
    READ_EXCEL_PARAMS = {
        'sheet_name': 0, # Read the first sheet
        'header': 0,     # Header is in the first row
    }

    # Colab File Upload (Parent Data)
    print("\n--- Uploading Parent Data (Parent.data.xlsx - Sheet1.csv) ---")
    uploaded_parent=0

    if not uploaded_parent:
        print("Parent file upload failed. Exiting.")
    else:
        parent_file_name = list(uploaded_parent.keys())[0]
        # FIX: Use pd.read_excel with BytesIO for robust Excel reading
        parent_df = pd.read_excel('parent.data.xlsx')
        parent_df = clean_column_names(parent_df)

        # Colab File Upload (Nanny Data)
        print("\n--- Uploading Nanny Data (Nanny.data.xlsx - Sheet1.csv) ---")
        uploaded_nanny = files.upload()

        if not uploaded_nanny:
            print("Nanny file upload failed. Exiting.")
        else:
            nanny_file_name = list(uploaded_nanny.keys())[0]
            # FIX: Use pd.read_excel with BytesIO for robust Excel reading
            nanny_df = pd.read_excel('nanny.data.xlsx')
            nanny_df = clean_column_names(nanny_df)

            print(f"\nSuccessfully loaded {len(parent_df)} Parents and {len(nanny_df)} Nannies.")
            print("-" * 50)

            # --- Perform Full Matching ---

            # These columns are now guaranteed to be clean due to clean_column_names()
            PARENT_NAME_COL = 'Invitee Name'
            NANNY_NAME_COL = 'Full Name'

            match_results = []

            # Iterate through every parent and every nanny to find all possible scores
            for parent_index, parent_row in parent_df.iterrows():
                for nanny_index, nanny_row in nanny_df.iterrows():

                    # Calculate the match score using the MVP function
                    final_score, component_scores = score_single_match(parent_row, nanny_row)

                    # Store all relevant details
                    result = {
                        'Parent Name': parent_row[PARENT_NAME_COL],
                        'Nanny Name': nanny_row[NANNY_NAME_COL],
                        'Final Score': final_score,
                    }
                    result.update(component_scores) # Add the individual component scores

                    match_results.append(result)

            # Create the results DataFrame
            results_df = pd.DataFrame(match_results)

            # --- Display Top Matches (MVP Scaling) ---

            # Sort by score in descending order
            top_matches = results_df.sort_values(by='Final Score', ascending=False)

            print(f"Total Matches Calculated: {len(top_matches)} (Every Parent vs. Every Nanny)")
            print("\n✅ TOP 5 MATCHES (Weighted Score Rank):")
            print("=" * 50)

            # Format and display
            display_cols = ['Parent Name', 'Nanny Name', 'Final Score', 'Location_Match', 'Language_Match', 'Availability_Match']
            print(top_matches[display_cols].head(5).to_markdown(index=False))
            print("=" * 50)

            print("\nNote: The 'Final Score' is the weighted average using the 30/20/20/15/15 structure.")
            print("To view all results, inspect the 'results_df' DataFrame.")















import pandas as pd
import io
import re # Import the regular expression module

# --- 1. WEIGHT DEFINITION (Copied from MVP) ---
WEIGHTS = {
    "Location_Match": 0.30,
    "Language_Match": 0.20,
    "Experience_Match": 0.20,
    "Availability_Match": 0.15,
    "Travel_Willingness": 0.15
}

# --- 2. SCORING LOGIC FUNCTIONS (Copied from MVP) ---

def score_location(parent_loc, nanny_pref_loc):
    """
    Component 1 (30%): Exact neighborhood match (100) or miss (0).
    """
    if str(parent_loc).strip().lower() == str(nanny_pref_loc).strip().lower():
        return 100
    return 0

def score_language(parent_lang, nanny_languages):
    """
    Component 2 (20%): If the Nanny speaks ANY of the Parent's required languages, it's a 100.
    """
    # Parent's language requirement is usually single (e.g., 'Tamil') or comma-separated.
    parent_langs = [l.strip().lower() for l in str(parent_lang).split(',')]
    # Nanny's language list might include proficiency (e.g., 'English (Basic)'). We strip the proficiency.
    nanny_langs = [l.split('(')[0].strip().lower() for l in str(nanny_languages).split(',')]

    for p_lang in parent_langs:
        if p_lang and p_lang in nanny_langs:
            return 100
    return 0

def score_experience(baby_age_str, nanny_exp_years):
    """
    Component 3 (20%): Scores Nanny experience against baby's age/needs.
    Rule: Requires minimal experience (0) for babies under 3 years, and 3+ years experience for older children.
    """
    try:
        # The 'np' (Not Provided) values will be NaN after loading, but we ensure robustness here.
        nanny_exp = float(nanny_exp_years)
    except (ValueError, TypeError):
        nanny_exp = 0.0

    baby_age_lower = str(baby_age_str).lower()

    # --- Robust Age Extraction Logic ---
    # Handle entries like "twin 1.5 years", "2 & 4 years", or just "6 months"

    # Extract all numbers (including decimals) from the age string
    numbers = re.findall(r'\d+\.?\d*', baby_age_lower)

    # Simple check for age based on keywords, defaulting to the first number found if any.
    if 'months' in baby_age_lower:
        # If age is in months, it is definitely < 3 years old.
        required_exp_years = 0
    elif numbers and 'years' in baby_age_lower:
        # If we found numbers and 'years', use the first number as the age for calculation
        try:
            baby_age_years = float(numbers[0])
            if baby_age_years < 3:
                required_exp_years = 0
            else:
                required_exp_years = 3
        except ValueError:
            # Fallback if number extraction fails unexpectedly
            required_exp_years = 3 # Assume older child if parsing is ambiguous
    else:
        # For ambiguous cases like 'np' or just 'twin', assume default requirement of 3+ years
        required_exp_years = 3
    # --- End Robust Age Extraction Logic ---

    # --- Scoring based on requirement ---

    if 'months' in baby_age_lower:
        # Check for babies 0-2 years (e.g., '6 months')
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 0:
        # Case: Age is < 3 years, minimal experience required
        if nanny_exp >= 0:
            return 100

    elif required_exp_years == 3:
        # Case: Age is >= 3 years (or complex age, defaulting to high need)
        if nanny_exp >= 3:
            return 100

    # Neutral score for complex cases, or where the rule is not a perfect match
    return 50

def score_availability(parent_type, nanny_hours_str):
    """
    Component 4 (15%): Scores based on type of nanny needed vs. hours available.
    """
    try:
        # Convert hours to float for comparison. 'np' values will fail gracefully to 0.
        nanny_hours_float = float(nanny_hours_str)
    except (ValueError, TypeError):
         nanny_hours_float = 0

    parent_type = str(parent_type).lower()

    if '24 hours' in parent_type:
        # 24hr live-in is a big ask. Assuming 20+ hours needed for a 100 score.
        if nanny_hours_float >= 20:
            return 100
        return 0 # Miss if not available for high hours

    elif 'full time' in parent_type or 'full time nanny' in parent_type:
        # Full time (6 to 10 hours) requires 10+ hours availability
        if nanny_hours_float >= 10:
            return 100
        return 40 # Partial miss if only available for less (e.g., 6 hours)

    elif 'part time' in parent_type or 'part time nanny' in parent_type:
        # Part time (0 to 6 hours) requires at least 6 hours
        if nanny_hours_float >= 6:
            return 100
        return 60 # Small miss if less than 6 hours

    return 20 # Low score for ambiguous types

def score_travel(travel_willingness):
    """
    Component 5 (15%): Scores based on Nanny's willingness to travel/commute.
    The column 'Willing to Travel (km)' in the data suggests a True/False value.
    """
    if str(travel_willingness).lower() == 'true':
        return 100
    return 50 # Not willing to travel is a constraint, hence 50

def score_single_match(parent, nanny):
    """
    Calculates the final weighted score for one Parent-Nanny pair (0-100).
    Returns: (final_score, component_scores_dict)
    """
    # 1. Calculate individual component scores (0-100)
    score_results = {
        "Location_Match": score_location(
            parent['Parent Location'],
            nanny['Parent Location'] # This column holds the Nanny's preferred work area
        ),
        "Language_Match": score_language(
            parent['Language'],
            nanny['Languages']
        ),
        "Experience_Match": score_experience(
            parent["Baby's Age"],
            nanny['Experience (Years)']
        ),
        "Availability_Match": score_availability(
            parent['Type of Nanny'],
            nanny['Working Hours']
        ),
        "Travel_Willingness": score_travel(
            nanny['Willing to Travel (km)']
        )
    }

    # 2. Apply weights and calculate final score
    final_score = 0
    for component, weight in WEIGHTS.items():
        component_score = score_results.get(component, 0)
        # Formula: Score * Weight
        weighted_contribution = component_score * weight
        final_score += weighted_contribution

    return round(final_score, 2), score_results


def clean_column_names(df):
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

# --- 3. EXECUTION BLOCK FOR COLAB ---

if __name__ == '__main__':
    print("Please upload the two EXCEL files when prompted:")

    # Define common robust read parameters
    # NOTE: read_excel is used instead of read_csv for robustness with Excel files.
    READ_EXCEL_PARAMS = {
        'sheet_name': 0, # Read the first sheet
        'header': 0,     # Header is in the first row
    }

    # Colab File Upload (Parent Data)
    print("\n--- Uploading Parent Data (Parent.data.xlsx - Sheet1.csv) ---")
    uploaded_parent =0

    if not uploaded_parent:
        print("Parent file upload failed. Exiting.")
    else:
        parent_file_name = list(uploaded_parent.keys())[0]
        # FIX: Use pd.read_excel with BytesIO for robust Excel reading
        parent_df = pd.read_excel('parent.data.xlsx')
        parent_df = clean_column_names(parent_df)

        # Colab File Upload (Nanny Data)
        print("\n--- Uploading Nanny Data (Nanny.data.xlsx - Sheet1.csv) ---")
        uploaded_nanny = files.upload()

        if not uploaded_nanny:
            print("Nanny file upload failed. Exiting.")
        else:
            nanny_file_name = list(uploaded_nanny.keys())[0]
            # FIX: Use pd.read_excel with BytesIO for robust Excel reading
            nanny_df = pd.read_excel('nanny.data.xlsx')
            nanny_df = clean_column_names(nanny_df)

            print(f"\nSuccessfully loaded {len(parent_df)} Parents and {len(nanny_df)} Nannies.")
            print("-" * 50)

            # --- Perform Full Matching ---

            # These columns are now guaranteed to be clean due to clean_column_names()
            PARENT_NAME_COL = 'Invitee Name'
            NANNY_NAME_COL = 'Full Name'

            match_results = []

            # Iterate through every parent and every nanny to find all possible scores
            for parent_index, parent_row in parent_df.iterrows():
                for nanny_index, nanny_row in nanny_df.iterrows():

                    # Calculate the match score using the MVP function
                    final_score, component_scores = score_single_match(parent_row, nanny_row)

                    # Store all relevant details
                    result = {
                        'Parent Name': parent_row[PARENT_NAME_COL],
                        'Nanny Name': nanny_row[NANNY_NAME_COL],
                        'Final Score': final_score,
                    }
                    result.update(component_scores) # Add the individual component scores

                    match_results.append(result)

            # Create the results DataFrame
            results_df = pd.DataFrame(match_results)

            # --- HOUR 3: TOP MATCHES PER PARENT (Updated for first 10 for Hour 4 testing) ---

            NUM_PARENTS_TO_SHOW = 10 # Changed from 3 to 10 for Hour 4 requirement
            TOP_K_MATCHES = 3

            print("\n\n--- HOUR 3 & 4: TOP 3 MATCHES FOR FIRST 10 PARENTS ---")

            # List to store scores only for the first 10 parents' matches
            scores_for_analysis = []

            # Iterate through the first ten parents in the parent DataFrame
            for i in range(NUM_PARENTS_TO_SHOW):
                if i >= len(parent_df):
                    break # Stop if there are fewer than 10 parents

                # Extract the name and location for the current parent
                current_parent_name = parent_df.iloc[i][PARENT_NAME_COL]
                current_parent_loc = parent_df.iloc[i]['Parent Location']

                # Filter results for this parent and sort to get the top matches
                parent_specific_matches = results_df[results_df['Parent Name'] == current_parent_name].sort_values(
                    by='Final Score',
                    ascending=False
                )

                # Store all scores for this parent for the statistics calculation
                scores_for_analysis.extend(parent_specific_matches['Final Score'].tolist())

                # Print the parent header line
                print(f"\nParent {i+1} ({current_parent_loc}): {current_parent_name}")

                # Print the top 3 matches in the requested format
                for _, match_row in parent_specific_matches.head(TOP_K_MATCHES).iterrows():
                    nanny_name = match_row['Nanny Name']
                    # Cast the score to an integer as shown in the example output
                    score = int(match_row['Final Score'])
                    print(f"  {nanny_name} ({score})")

            print("-" * 50)

            # --- HOUR 4: CALCULATE STATISTICS & SAVE RESULTS ---

            # 1. Convert list of scores to a Pandas Series for easy stats calculation
            scores_series = pd.Series(scores_for_analysis)

            # 2. Calculate Average, Min, Max scores
            avg_score = scores_series.mean()
            min_score = scores_series.min()
            max_score = scores_series.max()

            # 3. Count by Tier
            count_90_plus = len(scores_series[scores_series >= 90])
            count_70_89 = len(scores_series[(scores_series >= 70) & (scores_series < 90)])
            count_50_69 = len(scores_series[(scores_series >= 50) & (scores_series < 70)])
            count_less_50 = len(scores_series[scores_series < 50])

            print("\n\n--- HOUR 4: MATCH STATISTICS (First 10 Parents) ---")
            print(f"Total Matches Analyzed: {len(scores_series)}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Minimum Score: {min_score:.2f}")
            print(f"Maximum Score: {max_score:.2f}")
            print("-" * 35)
            print("Score Tier Counts:")
            print(f"  90+ Matches: {count_90_plus}")
            print(f"  70-89 Matches: {count_70_89}")
            print(f"  50-69 Matches: {count_50_69}")
            print(f"  <50 Matches: {count_less_50}")
            print("-" * 35)

            # 4. Save results to CSV (and enable download in Colab)
            output_filename = 'match_results_full_data.csv'
            results_df.to_csv(output_filename, index=False)
            files.download(output_filename)

            print(f"\n✅ Results saved to CSV: '{output_filename}' (download link below).")
            print("\n*** DAY 6 SUCCESS CHECKLIST ***")
            print(f"Average Score to post to channel: {avg_score:.2f}")
            print("-------------------------------")










import pandas as pd
import numpy as np

# Load your data
parents = pd.read_excel('Parent.data.xlsx')
nannies = pd.read_excel('Nanny.data.xlsx')

print(f"Total parents: {len(parents)}")
print(f"Total nannies: {len(nannies)}")

# Option 1: Random Split (70-30)
from sklearn.model_selection import train_test_split

parents_train, parents_test = train_test_split(
    parents, 
    test_size=0.25,  # 25% for testing
    random_state=42  # same split every time
)

nannies_train, nannies_test = train_test_split(
    nannies,
    test_size=0.25,
    random_state=42
)

print(f"\nTrain set: {len(parents_train)} parents, {len(nannies_train)} nannies")
print(f"Test set: {len(parents_test)} parents, {len(nannies_test)} nannies")

# Save the splits
parents_train.to_csv('parents_train.csv', index=False)
parents_test.to_csv('parents_test.csv', index=False)
nannies_train.to_csv('nannies_train.csv', index=False)
nannies_test.to_csv('nannies_test.csv', index=False)













import pandas as pd
import numpy as np

# Load your data
parents = pd.read_excel('Parent.data.xlsx')
nannies = pd.read_excel('Nanny.data.xlsx')

print(f"Total parents: {len(parents)}")
print(f"Total nannies: {len(nannies)}")

# Option 1: Random Split (70-30)
from sklearn.model_selection import train_test_split

parents_train, parents_test = train_test_split(
    parents,
    test_size=0.25,  # 25% for testing
    random_state=42  # same split every time
)

nannies_train, nannies_test = train_test_split(
    nannies,
    test_size=0.25,
    random_state=42
)


print(f"\nTrain set: {len(parents_train)} parents, {len(nannies_train)} nannies")
print(f"Test set: {len(parents_test)} parents, {len(nannies_test)} nannies")














import pandas as pd
import numpy as np
import re

# --- WEIGHT DEFINITION ---
WEIGHTS = {
    "Location_Match": 0.30,
    "Language_Match": 0.20,
    "Experience_Match": 0.20,
    "Availability_Match": 0.15,
    "Travel_Willingness": 0.15
}

# --- SCORING LOGIC FUNCTIONS ---

def score_location(parent_loc, nanny_pref_loc):
    """Exact neighborhood match (100) or miss (0)."""
    if str(parent_loc).strip().lower() == str(nanny_pref_loc).strip().lower():
        return 100
    return 0

def score_language(parent_lang, nanny_languages):
    """If nanny speaks any of parent's required languages → 100."""
    parent_langs = [l.strip().lower() for l in str(parent_lang).split(',')]
    nanny_langs = [l.split('(')[0].strip().lower() for l in str(nanny_languages).split(',')]
    for p_lang in parent_langs:
        if p_lang and p_lang in nanny_langs:
            return 100
    return 0

def score_experience(baby_age_str, nanny_exp_years):
    """Score nanny experience against baby's age."""
    try:
        nanny_exp = float(nanny_exp_years)
    except (ValueError, TypeError):
        nanny_exp = 0.0

    baby_age_lower = str(baby_age_str).lower()
    numbers = re.findall(r'\d+\.?\d*', baby_age_lower)

    if 'months' in baby_age_lower:
        required_exp_years = 0
    elif numbers and 'years' in baby_age_lower:
        try:
            baby_age_years = float(numbers[0])
            required_exp_years = 0 if baby_age_years < 3 else 3
        except ValueError:
            required_exp_years = 3
    else:
        required_exp_years = 3

    if 'months' in baby_age_lower or required_exp_years == 0:
        return 100 if nanny_exp >= 0 else 0
    elif required_exp_years == 3:
        return 100 if nanny_exp >= 3 else 50
    return 50

def score_availability(parent_type, nanny_hours_str):
    """Scores based on type of nanny needed vs hours available."""
    try:
        nanny_hours_float = float(nanny_hours_str)
    except (ValueError, TypeError):
        nanny_hours_float = 0

    parent_type = str(parent_type).lower()

    if '24 hours' in parent_type:
        return 100 if nanny_hours_float >= 20 else 0
    elif 'full time' in parent_type:
        return 100 if nanny_hours_float >= 10 else 40
    elif 'part time' in parent_type:
        return 100 if nanny_hours_float >= 6 else 60
    return 20

def score_travel(travel_willingness):
    """Score nanny’s willingness to travel/commute."""
    if str(travel_willingness).lower() == 'true':
        return 100
    return 50

# --- CORE MATCH FUNCTION ---

def calculate_match_score(parent, nanny):
    """Calculate overall match score using weighted components."""
    score_results = {
        "Location_Match": score_location(parent['Parent Location'], nanny['Parent Location']),
        "Language_Match": score_language(parent['Language'], nanny['Languages']),
        "Experience_Match": score_experience(parent["Baby's Age"], nanny['Experience (Years)']),
        "Availability_Match": score_availability(parent['Type of Nanny'], nanny['Working Hours']),
        "Travel_Willingness": score_travel(nanny['Willing to Travel (km)'])
    }

    final_score = sum(score_results[c] * WEIGHTS[c] for c in WEIGHTS)
    return round(final_score, 2)

# --- CLEAN COLUMN NAMES ---

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df


# --- MAIN EXECUTION ---

if __name__ == '__main__':
    print("🔹 Loading data from Excel files...")

    parents_train = pd.read_csv('parents_train.csv')
    nannies_train = pd.read_csv('nannies_train.csv')

    parents_train = clean_column_names(parents_train)
    nannies_train = clean_column_names(nannies_train)

    print(f"✅ Loaded {len(parents_train)} parents and {len(nannies_train)} nannies.\n")

    # TEST ON TRAINING DATA
    training_results = []
    for idx, parent in parents_train.iterrows():
        scores = []
        for idx2, nanny in nannies_train.iterrows():
            score = calculate_match_score(parent, nanny)
            scores.append({
                'parent': parent['Invitee Name'],
                'nanny': nanny['Full Name'],
                'score': score
            })

        # Get top 3 matches for each parent
        top_3 = sorted(scores, key=lambda x: x['score'], reverse=True)[:3]
        training_results.extend(top_3)

    # Calculate training metrics
    train_scores = [r['score'] for r in training_results]
    print(f"📊 Training Metrics:")
    print(f"  Average: {np.mean(train_scores):.2f}")
    print(f"  Min: {np.min(train_scores):.2f}")
    print(f"  Max: {np.max(train_scores):.2f}")
    print(f"  Std Dev: {np.std(train_scores):.2f}\n")

    # Display Top 5 Matches
    top_matches = sorted(training_results, key=lambda x: x['score'], reverse=True)[:5]
    print("🏆 TOP 5 MATCHES:")
    for m in top_matches:
        print(f"  👩 Parent: {m['parent']}  👶 Nanny: {m['nanny']}  → Score: {m['score']}")













# Load TEST data (algorithm has NEVER seen this)
parents_test = pd.read_csv('parents_test.csv')
nannies_test = pd.read_csv('nannies_test.csv')

# RUN SAME ALGORITHM (no changes!)
testing_results = []
for idx, parent in parents_test.iterrows():
    scores = []
    for idx2, nanny in nannies_test.iterrows():
        score = calculate_match_score(parent, nanny)  # SAME FUNCTION
        scores.append({
            'parent': parent['Invitee Name'],
            'nanny': nanny['Full Name'],
            'score': score
        })

    top_3 = sorted(scores, key=lambda x: x['score'], reverse=True)[:3]
    testing_results.extend(top_3)

# Calculate test metrics
test_scores = [r['score'] for r in testing_results]
print(f"Test Metrics:")
print(f"  Average: {np.mean(test_scores):.2f}")
print(f"  Min: {np.min(test_scores):.2f}")
print(f"  Max: {np.max(test_scores):.2f}")
print(f"  Std Dev: {np.std(test_scores):.2f}")

# COMPARE TRAIN vs TEST
print(f"\n📊 COMPARISON:")
print(f"Train avg: {np.mean(train_scores):.2f}")
print(f"Test avg:  {np.mean(test_scores):.2f}")
print(f"Difference: {np.mean(train_scores) - np.mean(test_scores):.2f}")

# If difference > 10 points = overfitting
if np.mean(train_scores) - np.mean(test_scores) > 10:
    print("⚠️ WARNING: Possible overfitting! Scores dropped on test data.")
else:
    print("✅ GOOD: Algorithm generalizes well to unseen data!")








import pandas as pd
import numpy as np

# ✅ Convert results list to DataFrame
results_df = pd.DataFrame(testing_results)

# ✅ Save detailed matching results
results_df.to_csv('day7_testing_results.csv', index=False)

# ✅ Create a summary metrics DataFrame
metrics = {
    'Train Avg': [np.mean(train_scores)],
    'Test Avg': [np.mean(test_scores)],
    'Difference': [np.mean(train_scores) - np.mean(test_scores)],
    'Overfitting Warning': [
        "⚠️ Yes" if np.mean(train_scores) - np.mean(test_scores) > 10 else "✅ No"
    ]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('day7_metrics.csv', index=False)

print("\n📁 Results saved successfully:")
print(" - day7_testing_results.csv (detailed matches)")
print(" - day7_metrics.csv (summary metrics)")











