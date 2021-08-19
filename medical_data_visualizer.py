import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

conditions = [
  (df['bmi'] > 25),
  (df['bmi'] <= 25)
]
values = [1, 0]
df['overweight'] = np.select(conditions, values)

df = df.drop(['bmi'], axis = 1)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = np.where(df['cholesterol'] > 1, 0, 1)
df['gluc'] = np.where(df['gluc'] > 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.DataFrame(df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()).rename(columns={'value': 'total'}).reset_index()

    # Draw the catplot with 'sns.catplot()'
    sns.set_theme(style='whitegrid')
    plot = sns.catplot(x='variable', y='total', data=df_cat, hue = 'value', kind= 'bar', col='cardio', ci=None)

    fig = plot.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    
    # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
   
    df2 = df[(df['ap_lo'] < df['ap_hi']) & 
             (df['height'] >= df['height'].quantile(0.025)) &
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))]
       

    # Calculate the correlation matrix
    df_corr = df2.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))



    # Set up the matplotlib figure
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    fig, ax = plt.subplots(figsize=(10,10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(df_corr, mask=mask, annot=True, fmt=".1f", cmap=cmap, vmin=-0.1, vmax=0.25, square=True, linewidths=0.1, cbar_kws={"shrink": .8}, ax=ax)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
