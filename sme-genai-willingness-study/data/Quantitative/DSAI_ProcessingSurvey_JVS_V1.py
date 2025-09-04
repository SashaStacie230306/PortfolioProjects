import textwrap
import itertools
import scipy as sc
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# To ensure no deprecated behaviour warning comes up
pd.set_option('future.no_silent_downcasting', True)

# Create pandas dataframe out of original survey csv
df = pd.read_csv("FINAL-ADSAI-knowledge_September 30, 2024_16.35.csv", skiprows=[1,2])

# Select columns corresponding to interesting questions for the research
df = df[["Q7","Q8","Q11","Q12","Q13","Q25","Q27","Q29"]]

# From the resized dataframe drop rows with empty values
df = df.dropna()

# Group departments into broader categories for analysis
rename_mapping = {
    "Administrative": "Management",
    "Executive & Management": "Management",
    "Operations": "Management",

    "Finance & Accounting": "Finance",

    "Legal": "Organizational Position",
    
    "IT & Technical Support": "IT",
    "Software Development": "IT",

    "Engineering": "Technology",
    "Research & Development": "Technology",
    
    "Customer Service": "Communication",

    "Human Resources": "Human Resource",
    
    "Sales & Marketing": "Sales & Marketing",
    "Creative & Design": "Sales & Marketing"
}

# Apply the new categories to the dataframe
df['Q29'] = df['Q29'].replace(rename_mapping)

# Drop the category with unkonwn department
df = df[df['Q29'] != 'Other']

# Rename columns to include general purpose of the question to make plotting easy
df.rename(columns={'Q7': 'Q7 Allowed to use ChatGPT',
                   'Q8': 'Q8 Use of GenAI at work',
                   'Q11': 'Q11 Concerns about using AI at work',
                   'Q12': 'Q12 Willingness to learn about AI tools',
                   'Q13': 'Q13 Weekly time willing to spend learning GenAI tools',
                   'Q25': 'Q25 Estimated reduction of errors by AI tools',
                   'Q27': 'Q27 Estimated contribution to job security',
                   'Q29': 'Company Department'}, inplace=True)


def get_plot_order(question_number:str):
    """
    Return the plot order of responses for a given question number. This helps 
    ensure that responses are plotted in a meaningful order, not alphabetically.
    
    Parameters:
    - question_number: The number of the question (as a string, e.g., '7').
    
    Returns:
    - List of responses in the desired order for plotting.
    """

    # Define all the right plotorders based on survey design
    Q7_order = ["Yes, I am allowed to use it without restrictions.",
            "Yes, but there are some restrictions or guidelines.",
            "No, it is banned at my workplace.",
            "I’m not sure."]

    Q8_order = ["Yes, I use them regularly for my work.",
            "Yes, I have used them occasionally for work.",
            "Rarely",
            "No, I do not use them at work."]


    Q11_order = ["Yes, I see significant issues or concerns with using AI in my role.",
            "Yes, I have some concerns, but I believe the benefits outweigh the risks.",
            "No, I don’t see any problems with using AI in my role.",
            "I’m not sure."]
    
    Q12_order = ["Not interested at all",
            "Not very interested",
            "Neutral",
            "Somewhat interested",
            "Very interested"]

    Q13_order = ["Less than 1 hour",
            "1-3 hours",
            "3-5 hours",
            "5-10 hours",
            "More than 10 hours",
            "I would prefer one-time learning dedication (not weekly recurring)"]

    Q25_order = ["Definitely not",
            "Probably not",
            "Might or might not",
            "Probably yes",
            "Definitely yes"]

    Q27_order = ["Definitely not",
            "Probably not",
            "Might or might not",
            "Probably yes",
            "Definitely yes"]

    # Assign the order to the (key) number of the question
    plot_order_dict = {
    "7": Q7_order,
    "8": Q8_order,
    "11": Q11_order,
    "12": Q12_order,
    "13": Q13_order,
    "25": Q25_order,
    "27": Q27_order}

    # Return the plot order for the given question string
    return plot_order_dict.get(question_number, None)


def create_department_barchart(column_name:str, df:pd.DataFrame, stacked=False):
    """
    Create either a grouped or 100% stacked bar chart for the given column 
    against company departments.
    
    Parameters:
    - column_name: The column name of the survey question to plot.
    - df: The dataframe containing survey data.
    - stacked: Boolean, whether to create a stacked chart (True for 100%).
    """
    # Custom color pallette by seaborn
    colors = sns.color_palette("CMRmap", df['Company Department'].nunique())

    # Create a cross-tabulation of the survey question against company departments
    cross_tab = pd.crosstab(df[column_name], df['Company Department'], dropna=False)

    # Reorder the rows of the cross-tab based on the expected response order
    cross_tab = cross_tab.reindex(get_plot_order(column_name.split(' ')[0][1:]), fill_value=0)
    
    # Plot a 100% stacked bar chart or a grouped bar chart based on the 'stacked' argument
    if stacked:
        # Convert to percentages for stacked bar chart
        cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
        ax = cross_tab_percentage.plot(kind='barh', stacked=True, color=colors)
        plt.xlim(0, 100)
    else:
        # Create a regular grouped bar chart
        ax = cross_tab.plot(kind='barh', stacked=False, color=colors)
        ax.legend_.remove()  # Remove the legend for grouped bar chart
    # Format y-axis labels to wrap text if labels are too long
    ax.set_yticklabels([textwrap.fill(label, width=30) if len(label) > 19 else label for label in cross_tab.index], fontsize=10)
    
    # Ensure x-axis tick marks are integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set x-axis label (Percentage for stacked bar, Count for grouped bar)
    plt.xlabel('Percentage' if stacked else 'Count', fontsize=10)
    
    # Set y-axis label to the survey question
    plt.ylabel(f'{column_name}', fontsize=10)

    # Set plot title to indicate the relationship between the survey question and company department
    plt.title(f'{column_name} response {"percentage" if stacked else "count"} by Department', fontsize=13, weight='bold')

    # Set a differnt location for the legend so it won't obstruct the plot, only for stacked because it won't be necessary twice when placed together.
    if stacked:
        ax.legend(title='Company Department', bbox_to_anchor=(1.05, 0.65), loc='upper left')

    # Save the plot as a PNG file
    plt.savefig(f'{"100pb" if stacked else "gbar"}_dep_vs_{column_name.split(" ")[0]}.png', bbox_inches='tight')


def get_num_val_map(question_number: str):
    """
    Returns a numeric value mapping for each response of the given question number.
    This is useful for converting categorical responses into numerical values 
    for statistical analysis like ANOVA or t-tests.
    
    Parameters:
    - question_number: The number of the question (as a string, e.g., '7').
    
    Returns:
    - A dictionary mapping responses to numerical values, or None if the question number is not found.
    """
    
    # Define response orders for each question as lists
    question_responses = {
        "7": ["Yes, I am allowed to use it without restrictions.",
              "Yes, but there are some restrictions or guidelines.",
              "No, it is banned at my workplace.",
              "I’m not sure."],

        "8": ["Yes, I use them regularly for my work.",
              "Yes, I have used them occasionally for work.",
              "Rarely",
              "No, I do not use them at work."],

        "11": ["Yes, I see significant issues or concerns with using AI in my role.",
               "Yes, I have some concerns, but I believe the benefits outweigh the risks.",
               "No, I don’t see any problems with using AI in my role.",
               "I’m not sure."],

        "12": ["Not interested at all",
               "Not very interested",
               "Neutral",
               "Somewhat interested",
               "Very interested"],

        "13": ["Less than 1 hour",
               "1-3 hours",
               "3-5 hours",
               "5-10 hours",
               "More than 10 hours",
               "I would prefer one-time learning dedication (not weekly recurring)"],

        "25": ["Definitely not",
               "Probably not",
               "Might or might not",
               "Probably yes",
               "Definitely yes"],

        "27": ["Definitely not",
               "Probably not",
               "Might or might not",
               "Probably yes",
               "Definitely yes"]
    }
    
    # Retrieve the response list for the given question number
    responses = question_responses.get(question_number, None)
    
    # If responses exist, map each response to a numeric value using enumerate
    if responses is None:
        return None
    
    # Return the numeric mapping (e.g., {'Definitely yes': 4})
    return {response: i for i, response in enumerate(responses)}



def pairwise_t_tests(df:pd.DataFrame, column_name:str, alpha=0.05):
    """
    Perform pairwise t-tests between departments on the given column and apply 
    Bonferroni correction to account for multiple comparisons.
    
    Parameters:
    - df: The DataFrame containing survey data.
    - column_name: The name of the column on which to perform t-tests.
    - alpha: The significance level, default is 0.05.
    
    Prints the results of significant comparisons.
    """
    count = 0

    # Convert the column values to numeric, setting errors to NaN if conversion fails
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # Drop rows with missing or non-numeric values in the selected column
    df = df.dropna(subset=[column_name])

    # Get unique departments from the 'Company Department' column
    departments = df["Company Department"].unique()

    # Generate all unique pairs of departments for pairwise comparison
    department_pairs = list(itertools.combinations(departments, 2))
    
    # Apply Bonferroni correction to alpha for multiple comparisons
    alpha_bfcorrected = alpha / len(department_pairs)

    # Perform pairwise t-tests between departments
    for dep1, dep2 in department_pairs:
        # Extract data for the two departments
        data1 = df[df["Company Department"] == dep1][column_name]
        data2 = df[df["Company Department"] == dep2][column_name]

        # Perform a t-test assuming equal variances as tested in ANOVA
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=True)
        
        # If p-value is less than the corrected alpha, print significant results
        if p_val < alpha_bfcorrected:
            print(f'Difference between departments significant (p < {alpha_bfcorrected:.4f})\n{dep1} VS {dep2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}\n')
            count += 1
        else:
            print(f'Difference between departments insignificant (p > {alpha_bfcorrected:.4f})\n{dep1} VS {dep2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}\n')
    # If no significant differences were found
    if count == 0:
        print("No significant differences found between individual departments")


def perform_statistical_analysis_by_department(column_name:str, df:pd.DataFrame):
    """
    Perform statistical analysis to test for differences between company departments
    on a given survey question. This function includes:
    
    1. **Levene’s Test**: Checks for equality of variances between departments.
       - If the p-value of Levene's test is greater than 0.05, standard ANOVA is performed.
       - If the p-value of Levene’s test is less than 0.05, the function halts ANOVA 
         (indicating unequal variances) and suggests ANOVA may not be appropriate.
    
    2. **ANOVA (Analysis of Variance)**: Compares means across departments.
       - If the ANOVA p-value is less than 0.05, pairwise t-tests are conducted to find 
         which department pairs differ significantly.
    
    3. **Pairwise T-Tests**: Tests for significant differences between each pair of departments.
       - Bonferroni correction is applied to adjust for multiple comparisons.
    
    Parameters:
    - column_name: The survey question (column) to analyze.
    - df: The dataframe containing survey data.

    Prints:
    - Results of Levene's test.
    - ANOVA results if variances are equal.
    - Pairwise t-test results if ANOVA finds significant differences.
    - A message indicating unequal variances if Levene’s test fails.
    """
    # Replace categorical responses with their corresponding numeric values
    df[column_name] = df[column_name].replace(get_num_val_map(column_name.split(' ')[0][1:]))
    
    # Group data by company department and extract responses for ANOVA
    department_data = [group[column_name].values for name, group in df.groupby("Company Department")]

    # Perform Levene's test to check for equal variances between groups
    stat, p_levene = sc.stats.levene(*department_data)

    # Print the question number for reference
    print(f"---------{column_name.split(' ')[0]}---------")
    
    # If variances are equal (p > 0.05), perform standard ANOVA
    if p_levene > 0.05:
        print(f"Levene’s test passed (p={p_levene:.3f}), variances are equal.\n")
        
        # Perform ANOVA across the departments
        f_stat, p_val = sc.stats.f_oneway(*department_data)
        print(f'ANOVA: {column_name} vs Department: f value= {f_stat:.3f}, p-value= {p_val:.3f}\n')
        
        # If ANOVA is significant (p < 0.05), proceed with pairwise t-tests
        if p_val < 0.05:
            print("Significant difference found to reject H0, proceeding with Pairwise T-test between departments:")
            pairwise_t_tests(df, column_name)
            print("\n\n")
        else:
            print("No significant difference found, cannot reject H0\n\n")
    # If variances are not equal (p < 0.05), ANOVA may not be valid
    else:
        print(f"{column_name.split(' ')[0]} Levene’s test failed (p={p_levene:.3f}), variances are not equal. ANOVA may not be appropriate.\n\n")


# Loop over every column_name except the last column: Company Department
for column_name in df.columns[:-1]:
    # Create a 100% stacked barchart
    create_department_barchart(column_name, df, stacked=True)
    # Create a grouped barchart
    create_department_barchart(column_name, df, stacked=False)
    # Perform ANOVA and pairwise t-test when applicable 
    perform_statistical_analysis_by_department(column_name, df)

# Save the processed DataFrame to a CSV file without including the row indices
df.to_csv("procd_survey_data_JVS.csv", index=False)