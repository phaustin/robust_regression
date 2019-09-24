# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   toc:
#     base_numbering: 1
#     nav_menu: {}
#     number_sections: true
#     sideBar: true
#     skip_h1_title: false
#     title_cell: Table of Contents
#     title_sidebar: Contents
#     toc_cell: true
#     toc_position: {}
#     toc_section_display: true
#     toc_window_display: true
# ---

# %% [markdown] {"toc": true}
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# %%
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import pymc3 as pm
the_file = "interaction_metric_two_canyons.csv"
df=pd.read_csv(the_file)
df['x']=df['x'] - 1

# %%
fig, ax = plt.subplots(figsize=(7,7))
ax.errorbar(df['x'].values, df['y'].values, fmt='ro', 
              yerr=df['y_error'].values,xerr=df['x_error'].values,ecolor='black');

# %%
with pm.Model() as model_robust:
    family = pm.glm.families.StudentT()
    pm.glm.GLM.from_formula('y ~ x', df, family=family)
    trace_robust = pm.sample(40000, cores=2)
   

# %% {"scrolled": false}
pm.plot_trace(trace_robust);

# %%
fig=plt.figure(figsize=(10,7))
pm.plot_posterior_predictive_glm(trace_robust,
                                 label='posterior predictive regression lines')
ax=fig.axes[0]
ax.errorbar(df['x'].values, df['y'].values, fmt='ro', 
              yerr=df['y_error'].values,xerr=df['x_error'].values,ecolor='black');

# %%
