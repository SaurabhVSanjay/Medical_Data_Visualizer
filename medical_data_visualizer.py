import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("C:\\Users\\vssau\\medical_examination.csv")

# 2
bmi = df['weight']/((df['height']/100)**2)
df['overweight'] = bmi.apply(lambda x: 1 if x> 25 else 0)

# 3
def transform_value(value):
    if value == 1:
        return 0
    elif value > 1:
        return 1
    else:
        return value

df['cholesterol'] = df['cholesterol'].apply(transform_value)
df['gluc'] = df['gluc'].apply(transform_value)

# 4
def draw_cat_plot():
    # 5
    # 5
    categorical_columns = ['cardio', 'active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    df[categorical_columns] = df[categorical_columns].astype(str)
    df_cat = df[categorical_columns]
    #df_cat = df.melt(id_vars=['cardio'], value_vars=categorical_columns, var_name='feature', value_name='value')

   ## Plot using seaborn's catplot
   #g = sns.catplot(x='feature', hue='value', col='cardio', kind='count', data=df_cat, height=5, aspect=1.5)
   #g.set_axis_labels('Feature', 'Count')
   #g.despine(left=True)
   #plt.show()
#

    # 6
    df_cat = pd.melt(df_cat, id_vars=['cardio'], value_vars=categorical_columns, var_name='feature', value_name='value')
    

    # 7
    g = sns.catplot(x='feature', hue='value', col='cardio', kind='count', data=df_cat, height=5, aspect=1.5)
    g.set_axis_labels('variable', 'total')
    g.set_titles('Cardio = {col_name}')
    g.despine(left=True)


    # 8
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(corr, mask=mask, cmap='inferno', annot=True, fmt='.1f', annot_kws={'size': 8}, ax=ax, cbar_kws={'shrink': .6})

    # 16
    fig.savefig('heatmap.png')
    return fig

# Execute the functions to create and save the plots
#draw_cat_plot()
#draw_heat_map()
