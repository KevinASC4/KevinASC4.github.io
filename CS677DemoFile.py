# %% [markdown]
# # Seaborn Exercises
# 
# * This homework is designed to test your skills at creating data visualizations using Seaborn.
# * There are **FOUR** tasks. Solve all four. 
# * You are highly encouraged to refer to the official Seaborn documentation (apart from class notes) while solving this homework.
# * Link to Seaborn documentation: https://seaborn.pydata.org/
# * Additional resources: MatplotLib tutorial in Sec. 1.4 in the Scientific Python lectures (link available on the course website under "Syllabus and General Information")

# %% [markdown]
# ## Imports
# 
# Run the cell below to import the libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# %% [markdown]
# ## The Data
# 
# The dataset for this homework is also available on Kaggle: https://www.kaggle.com/rikdifos/credit-card-approval-prediction
# 
# Brief Description:
# 
# - Credit score cards are a common risk control method in the financial industry. 
# - It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings.
# - The bank is able to decide whether to issue a credit card to the applicant.
# - Essentially, credit scores can help quantify the magnitude of risk.

# %% [markdown]
# Feature Information:
# 
# <table>
# <thead>
# <tr>
# <th>application_record.csv</th>
# <th></th>
# <th></th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>Feature name</td>
# <td>Explanation</td>
# <td>Remarks</td>
# </tr>
# <tr>
# <td><code>ID</code></td>
# <td>Client number</td>
# <td></td>
# </tr>
# <tr>
# <td><code>CODE_GENDER</code></td>
# <td>Gender</td>
# <td></td>
# </tr>
# <tr>
# <td><code>FLAG_OWN_CAR</code></td>
# <td>Is there a car</td>
# <td></td>
# </tr>
# <tr>
# <td><code>FLAG_OWN_REALTY</code></td>
# <td>Is there a property</td>
# <td></td>
# </tr>
# <tr>
# <td><code>CNT_CHILDREN</code></td>
# <td>Number of children</td>
# <td></td>
# </tr>
# <tr>
# <td><code>AMT_INCOME_TOTAL</code></td>
# <td>Annual income</td>
# <td></td>
# </tr>
# <tr>
# <td><code>NAME_INCOME_TYPE</code></td>
# <td>Income category</td>
# <td></td>
# </tr>
# <tr>
# <td><code>NAME_EDUCATION_TYPE</code></td>
# <td>Education level</td>
# <td></td>
# </tr>
# <tr>
# <td><code>NAME_FAMILY_STATUS</code></td>
# <td>Marital status</td>
# <td></td>
# </tr>
# <tr>
# <td><code>NAME_HOUSING_TYPE</code></td>
# <td>Way of living</td>
# <td></td>
# </tr>
# <tr>
# <td><code>DAYS_BIRTH</code></td>
# <td>Birthday</td>
# <td>Count backwards from current day (0), -1 means yesterday</td>
# </tr>
# <tr>
# <td><code>DAYS_EMPLOYED</code></td>
# <td>Start date  of employment</td>
# <td>Count backwards from current day(0). If  positive, it means the person currently unemployed.</td>
# </tr>
# <tr>
# <td><code>FLAG_MOBIL</code></td>
# <td>Is there a mobile   phone</td>
# <td></td>
# </tr>
# <tr>
# <td><code>FLAG_WORK_PHONE</code></td>
# <td>Is there a work phone</td>
# <td></td>
# </tr>
# <tr>
# <td><code>FLAG_PHONE</code></td>
# <td>Is there a phone</td>
# <td></td>
# </tr>
# <tr>
# <td><code>FLAG_EMAIL</code></td>
# <td>Is there an email</td>
# <td></td>
# </tr>
# <tr>
# <td><code>OCCUPATION_TYPE</code></td>
# <td>Occupation</td>
# <td></td>
# </tr>
# <tr>
# <td><code>CNT_FAM_MEMBERS</code></td>
# <td>Family size</td>
# <td></td>
# </tr>
# </tbody>
# </table>

# %%
df = pd.read_csv('application_record.csv')

# %%
df.head(3)

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ## TASKS 
# 
# ### Recreate the plots shown in the markdown image cells. 
# 
# Each plot also contains a brief description of what it is trying to convey. 
# 
# Note, these are meant to be quite challenging. Start by first replicating the most basic form of the plot, then attempt to adjust its styling and parameters to match the given image.
# 
# Closer reproductions will receive higher scores.
# 
# **NOTE**: You may need to perform extra calculations on the Pandas DataFrame before calling Seaborn to create the plot.
# 
# ---

# %% [markdown]
# ### TASK: Recreate the Scatter Plot shown below
# 
# * The scatterplot attempts to show the relationship between the days employed versus the age of the person (DAYS_BIRTH) for people who were not unemployed.
# * Note, to reproduce this chart you must **remove unemployed people** from the dataset first.
# * Also note the **sign** of the axis, they are both transformed to be positive.
# * What do the `alpha` and `linewidth` parameters in a Seaborn scatterplot control? Since there are so many points stacked on top of each other, feel free to modify these two parameters to present a clear plot.
# 
# <figure>
# <img src="task_one.jpg">
# </figure>

# %%
# CODE HERE TO RECREATE THE PLOT SHOWN ABOVE
df_employed = df[df['DAYS_EMPLOYED']<= 2000 ]  # Remove the outlier
# Transform the DAYS_BIRTH and DAYS_EMPLOYED to positive using abs()
df_employed['DAYS_BIRTH'] = df_employed['DAYS_BIRTH'].abs()  # Make age positive
df_employed['DAYS_EMPLOYED'] = df_employed['DAYS_EMPLOYED'].abs()  # Make days employed positive

# Create the scatter plot
plt.figure(figsize=(7, 7))
sns.scatterplot(data=df_employed, x='DAYS_BIRTH', y='DAYS_EMPLOYED', alpha=.1, linewidth=.2)

# Calculate the IQR for DAYS_BIRTH
Q1_birth = df_employed['DAYS_BIRTH'].quantile(0.25)
Q3_birth = df_employed['DAYS_BIRTH'].quantile(0.75)
IQR_birth = Q3_birth - Q1_birth

# Calculate the IQR for DAYS_EMPLOYED
Q1_employed = df_employed['DAYS_EMPLOYED'].quantile(0.25)
Q3_employed = df_employed['DAYS_EMPLOYED'].quantile(0.75)
IQR_employed = Q3_employed - Q1_employed

# Define bounds for removing outliers
lower_bound_birth = Q1_birth - 1.5 * IQR_birth
upper_bound_birth = Q3_birth + 1.5 * IQR_birth
lower_bound_employed = Q1_employed - 1.5 * IQR_employed
upper_bound_employed = Q3_employed + 1.5 * IQR_employed

# Filter out the outliers
df_filtered = df_employed[
    (df_employed['DAYS_BIRTH'] >= lower_bound_birth) & 
    (df_employed['DAYS_BIRTH'] <= upper_bound_birth) &
    (df_employed['DAYS_EMPLOYED'] >= lower_bound_employed) & 
    (df_employed['DAYS_EMPLOYED'] <= upper_bound_employed)
]
# Set labels and title
plt.xlabel('DAYS_BIRTH')
plt.ylabel('DAYS_EMPLOYED')


# Show the plot
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ### TASK: Recreate the Distribution Plot shown below:
# <figure>
# <img src="DistPlot_solution.png">
# </figure>
# 
# Note, you will need to figure out how to calculate "Age in Years" from one of the columns in the DF. Think carefully about this.

# %%
# CODE HERE TO RECREATE THE PLOT SHOWN ABOVE
# Create the distribution plot

plt.figure(figsize=(15, 3))
# Calculate age in years
df['Age_Years'] = df['DAYS_BIRTH'].abs() / 365  # Convert DAYS_BIRTH to positive years
# Set labels and title
sns.histplot(df['Age_Years'], color='red')

# Show the plot
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ### TASK: Recreate the Categorical Plot shown below:
# <figure>
# <img src='catplot_solution.png'>
# </figure>
# 
# - This plot shows information only for the **bottom half** of income earners in the data set.
# - It shows the boxplots for each category of NAME_FAMILY_STATUS column for displaying their distribution of their total income.
# - Note: You will need to adjust or only take part of the dataframe *before* recreating this plot.
# - You may want to explore the `order` parameter to get the xticks in the exact order shown here

# %%
# Code here
# Calculate the median total income
median_income = df['AMT_INCOME_TOTAL'].median()

# Filter the DataFrame for the bottom half of income earners
df_bottom_half = df[df['AMT_INCOME_TOTAL'] <= median_income]
# Determine the order of NAME_FAMILY_STATUS based on total income for the bottom half
order = df_bottom_half.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].median().sort_values().index

# Create the boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_bottom_half, x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL', hue='FLAG_OWN_REALTY', order=order)
# Remove labels
plt.xlabel('')
plt.ylabel('')
plt.title('')  # Optionally remove the title as well

# Position the legend on the side
plt.legend(title='FLAG_OWN_REALTY', loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# # Heatmaps 
# 
# In Seaborn, **heatmaps** are used to **visualize data in a matrix format**, where the individual values in the matrix are represented as **colored cells**. Heatmaps are especially useful for showing the magnitude of data and patterns across a 2D space. 
# 
# Each cell in the heatmap is colored based on the value it contains, with a color gradient that reflects the intensity or magnitude of the value.
# 
# Seaborn Heatmap documentation: https://seaborn.pydata.org/generated/seaborn.heatmap.html
# 

# %% [markdown]
# ### What Heatmaps Show:
# 1. **Matrix of values**: Heatmaps visualize data that is organized in a grid (e.g., a correlation matrix, confusion matrix, or any two-dimensional data).
# 2. **Color-coded magnitude**: The color of each cell in the heatmap reflects the **magnitude** of the corresponding data point. Lighter or darker colors represent higher or lower values, depending on the color map.
# 3. **Patterns and relationships**: Heatmaps are excellent for identifying **patterns, trends**, or **correlations** between variables by using colors to highlight significant values or clusters in the data.

# %% [markdown]
# ### Common Use Cases for Heatmaps:
# - **Correlation matrix**: A heatmap is often used to visualize a **correlation matrix**, where each cell represents the correlation coefficient between two variables.
# - **Confusion matrix**: It is also commonly used to display a **confusion matrix** in classification tasks, where the color shows how often different categories are confused with each other. (We will see this later in the course)
# - **Clustered data**: Heatmaps can highlight patterns in clustered data, making it easier to see relationships between features or groups.
# 

# %% [markdown]
# 
# ### TASK: Recreate the Heat Map shown below:
# <figure>
# <img src='heatmap_solution.png', width='450'>
# </figure>

# %% [markdown]
# 
# 
# * This heatmap shows the correlation between the columns in the dataframe. You can get correlation with `.corr()`
# * Note here, that the `FLAG_MOBIL` column has `NaN` correlation with every other column, so you should drop it before calling `.corr().`
# * Finally, what `cmap` value will give you this colormap? Choose from the MatplotLib color palette [available here](https://matplotlib.org/stable/users/explain/colors/colormaps.html).

# %%
# CODE HERE
df_hm = df.drop(columns=['FLAG_MOBIL'])

# Select only numeric columns
numeric_columns = df_hm.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Optionally, drop rows and columns with NaN values from the correlation matrix
correlation_matrix = correlation_matrix.dropna().dropna(axis=1)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='viridis', square=True, cbar=False)
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.title('')


# %%


# %%


# %%



