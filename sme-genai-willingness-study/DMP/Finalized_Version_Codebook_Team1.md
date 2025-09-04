---
title: "Codebook Group 1 Data Science & AI"
author: "Kudyba, B., Pitulice, T., Van Santen, J.C.A., Stacie, S. and Vicheva, V."
date: "27/09/2024"
---

## Project Description

This research examines the factors influencing employees' willingness to learn generative AI tools in SMEs, focusing on departments, sector, education level, and demographics.

##Study design and data processing

The aim is to obtain industry insights from conducting an online survey through
’Prolific.com’. This platform offers a quick and effective solution to directly target the SME employees (population)
who are active on Prolific (sample). After analysing the data from the survey, interviews will be conducted with SME
employees to look for specific reasons as to why differences were present or not.


## Data Collection and Clearning
###Collection of the raw data
Data is to be collected from a conducted online survey through ’Prolific.com'. Participants self-reported on their willingness to learn about generative AI, SME size, and other relevant factors.

###Notes on the original (raw) data 
This section here is not applicable at the moment because the survey has not been conducted yet; all data will be hypothetical until the actual responses are gathered.

##Creating the tidy datafile

###Guide to create the tidy data file
Description on how to create the tidy data file (1. download the data, ...)/

###Cleaning of the data
The description is not applicale at the moment. However,, talking hypothetically in the cleaning script our team will remove incomplete responses, recodes SME size categories into uniform intervals, and standardizes responses for willingness to learn about AI.

## Variables
##Description of the variables in the tiny_data.txt file
General description of the file including:
1. Dimensions: [To be determined] rows and [X columns].
   
2. Summary: The dataset will contain employees’ responses to demographic questions, willingness to learn AI, current knowledge of AI tools, and perceived barriers.
   
3. Variables Present: SME size, sector, department, investment in technology, revenue, willingness to learn AI, and more.


###Variable 1 (repeat this section for all variables in the dataset)
Short description in bullet points of what the variable describes including but not limited to:
**Because the survey is still not published in ’Prolific.com' the following variables are only **hypothetical ones**. We will finalize them within the end of the research. 

Variable 1: SME Size
Name: sme_size
Description: Size of the company in terms of number of employees.
Measurement: Ordinal (1–10, 11–50, 51–250).
Class: Numeric
Unique Values: 1 = 1–10, 2 = 11–50, 3 = 51–250
Unit of Measurement: Number of employees.
Notes on Variable 1:
Hypothetical values will be assigned based on employee responses.

Variable 2: Willingness to Learn about AI
Name: willingness_ai
Description: Respondent’s willingness to learn AI on a 5-point Likert scale.
Measurement: Ordinal (Strongly Disagree to Strongly Agree).
Class: Numeric
Unique Values: 1 = Strongly Disagree, 5 = Strongly Agree
Notes on Variable 2:
Will be derived from employee responses to question 1.11.

Variable 3: Company Sector
Name: company_sector
Description: The sector in which the respondent's company operates.
Measurement: Nominal.
Class: String
Unique Values: Technology & IT, Healthcare, Finance, Manufacturing, Retail, Energy

####Notes on variable 1
If available, some additional notes on the variable not covered elsewehere. If no notes are present leave this section out.

##Sources
No external sources were used as the survey is hypothetical at this stage.

##Annex
Additional details will be added after the survey is conducted and data is collected.
## Variables

### Victoria Vicheva’s Paper -Study on AI Perception in Marketing and Media SMEs (Qualitative Analysis)
1. **SME Size**: Indicates the company size (small or medium) for comparative purposes.
2. **Age**: Age range of employees to analyze potential generational differences in AI perspectives.
3. **Willingness to Learn AI**: Measures employees' openness to adopting AI tools on a Likert scale.
4. **Perceived Job Threat vs. Growth Opportunity**: Evaluates if employees view AI adoption as a career growth enhancer or as a potential job threat.
5. **Department**: Specifies if employees belong to Marketing or Media sectors to capture sector-specific perspectives.

### Bartosz Kudyba’s Paper - Sectoral Comparison of Willingness to Learn AI (Quantitative Analysis)
1. **Sector**: Categorical variable differentiating sectors like finance, retail, and customer service.
2. **Willingness to Learn AI**: Likert scale measuring enthusiasm or openness toward AI adoption across sectors.
3. **Frequency of AI Use at Work**: Tracks self-reported regularity of AI tool usage.
4. **Barriers to AI Learning**: Identifies common challenges in AI learning across sectors (e.g., limited resources).

### Tudor Pitulice’s Paper - Exploration of Generational Gaps in AI Usage (Quantitative Analysis)
1. **Generation**: Categorical variable grouping respondents by Gen Z, Millennials, and Gen X.
2. **AI Usage Frequency**: Measures AI use frequency in daily life and work environment.
3. **Attitudes toward AI**: Evaluates perceived benefits and challenges across different generations.
4. **Willingness to Learn AI**: Likert scale indicating employees' openness toward adopting AI.

### Jonathan van Santen’s Paper - Departmental Differences in AI Learning Willingness (Quantitative Analysis)
1. **SME Department**: Categorical variable for department grouping (e.g., IT, Communications, HR).
2. **Use of GenAI**: Tracks generative AI usage frequency within departments.
3. **Concerns about AI**: Measures apprehensions and perceived risks associated with AI.
4. **Weekly Learning Time Commitment**: Self-reported willingness to allocate time toward learning AI.
5. **Perceived Error Reduction**: Departmental perspectives on AI's effectiveness in reducing workplace errors.

### Sasha Stacie's Paper -on Educational Influence on AI Willingness in SMEs (Quantitative Analysis)
1. **Educational Level**: Categorical variable based on highest education completed (high school, bachelor’s, master’s).
2. **Willingness Score**: Composite score combining interest in AI learning with time commitment, indicating the overall willingness to engage with AI tools.
3. **Interest in Learning AI**: Likert scale measuring employees' enthusiasm for learning AI.
4. **Time Commitment for Learning AI**: Weekly hours employees are willing to dedicate to learning AI.
