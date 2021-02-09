#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas_datareader.data as reader
import datetime as dt
import numpy
import scipy.stats


# In[2]:


pd.options.mode.chained_assignment = None


# In[3]:


df = pd.read_csv("stock_data.csv")  # load stock data


# In[4]:


df


# In[5]:


df.info() #screen the data and check the data types


# In[6]:


# We see above that RET is not numerical.

print(df.RET.describe()) # check the values of RET


# In[7]:


print(df.RET.value_counts())  # check the values of RET


# In[8]:


df = df[df.RET!="C"] # remove observations with invalid data


# In[9]:


df.RET = df.RET.astype("float") # change type to float for calculations


# In[10]:


df.date = pd.to_datetime(df.date) # convert type of object to datetime


# In[11]:


df.info()


# In[12]:


df.PERMNO.nunique()


# In[13]:


df.CUSIP.nunique()


# In[14]:


df = df.drop(["SHRCD", "EXCHCD", "CUSIP", "FACPR", "FACSHR"], axis=1) #exclude unnecessary columns


# In[15]:


df


# In[16]:


# remove duplicates

df.drop_duplicates(subset=["PERMNO", "date"], inplace=True)


# In[17]:


df # 50 observations are removed


# In[18]:


#lagged price and lagged share numbers to calculate lagged market cap

df["PRC_shifted"] = df.PRC.shift(1)

df["SHROUT_shifted"] = df.SHROUT.shift(1)


# In[19]:


df


# In[20]:


# find first occurrence of assets to remove those observations

indexes=[]

for i in df.PERMNO.unique():
    indexes.append(df.loc[df.PERMNO==i, "date"].idxmin())


# In[21]:


# indexes to drop
indexes


# In[22]:


df.drop(indexes, inplace=True) # remove first occurrences of assets


# In[23]:


df


# In[24]:


df["MarketCap"] = df.PRC_shifted * df.SHROUT_shifted  # add market capitalization column for lagged market caps


# In[25]:


df


# In[26]:


end = dt.date(2020,9,30)
start = dt.date(2018,1,31)


# In[27]:


#Load size cut offs

ME = reader.DataReader("ME_Breakpoints", "famafrench", start, end)

ME = ME[0]

ME.drop("Count", axis = 1, inplace=True)

ME.head()


# In[28]:


label= ["P"+str(i) for i in range(1,21)]   #portfolio names

for i in df.date.unique():  #loop over months
    
    bins = ME.loc[i]        #take cut off values from ME
    
    bins = [i*1000 for i in bins]   #multiply ME cut-off values with 1000 to match the market cap scale in stock data
    
    bins.insert(0,0)        #insert 0 as beginning cut off value
    
    df.loc[df.date==i, "Portfolio"] =  pd.cut(df.loc[df.date==i, "MarketCap"], bins=bins, labels=label)
    #group data frame with size cut-off values and assign a portfolio category to each asset


# In[29]:


df


# In[30]:


#check if all assets are within the size boundaries

df.isnull().sum()


# In[31]:


df.loc[df.Portfolio.isnull(), "Portfolio"] = "P20"  #place 106 XL stocks/rows in Portfolio 20


# In[32]:


df.isnull().sum() # all stocks are categorized


# In[33]:


#save portfolio total caps of monthly re-balanced portfolios to use when calculating weights
p_caps = df.groupby(["date", "Portfolio"])["MarketCap"].sum().reset_index().rename(columns={"MarketCap":"PortfolioCap"})


#add portfolio caps to the same data frame
df_merged = pd.merge(df, p_caps, how="left", on=["date", "Portfolio"])


#calculate weighted return of each asset
df_merged["W_Return"] = df_merged.RET * df_merged.MarketCap / df_merged.PortfolioCap


#aggregate weighted returns to calculate total portfolio return
df_1 = df_merged.groupby(["date", "Portfolio"])["W_Return"].sum().reset_index().rename(columns={"W_Return":"P_Return"})


#percentage
df_1["P_Return"] = df_1["P_Return"]*100


#add portfolio categories to the same dataframe
df_2= pd.merge(df_merged, df_1, on=["date", "Portfolio"])

#create a new data frame with monthly portfolio returns
portfolio_matrix = pd.pivot_table(data=df_2, values="P_Return", index="date", columns="Portfolio")

portfolio_matrix = portfolio_matrix.reindex(columns=label)

portfolio_matrix


# In[34]:


portfolio_matrix.describe() #summary


# In[35]:


# load Fama French data to get rf rate

FF = reader.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)
FF = FF[0]
FF.head()


# In[36]:


FF = FF[1:] # remove the first row
FF.head()


# In[37]:


md = pd.read_csv("market_return.csv") # load market return data
md.head()


# In[38]:


md.drop(0, inplace=True) # remove the first row


# In[39]:


md.head()


# In[40]:


portfolio_matrix.index= md.index  #set indexes of data frames equal
df_port = portfolio_matrix.join(md) # join portfolio return matrix and market returns
df_port.head()


# In[41]:


df_port.index = FF.index  #set indexes equal
df_port.totval = FF.RF    #replace totval column with RF rates from fama french data
df_port = df_port.rename(columns={'totval': 'RF'}) # rename the column as RF
df_port.vwretd = df_port.vwretd*100 # multiply market return with 100 as other returns are in percentage
df_port


# In[42]:


df_famamacbeth = df_port.copy(deep=True) # take a copy of returns to use in Fama Macbeth test in task 2


# In[43]:


# subtract risk free rate from each portfolio return and obtain portfolio excess returns

for p in label:
    df_port[p] = df_port[p] - df_port['RF']
    
df_port.head()


# In[44]:


#market excess return

df_port["MktRF"] = df_port["vwretd"] - df_port["RF"]


# In[45]:


df_port
df = df_port.drop(["DATE"], axis = 1)


# In[46]:


df.head()


# In[47]:


df.to_excel("port.xlsx")


# In[48]:


df.to_csv('port.csv')


# In[49]:


#running time-series regressions to estimate portfolio betas, alphas, t stats and R2s

import statsmodels.formula.api as sm

alphas = []
betas = []
t_alpha = []
t_beta = []
r_squared = []

for p in label:
    formula = p + " ~ MktRF"
    model = sm.ols(formula, data=df).fit()
    alphas.append(model.params[0])
    betas.append(model.params[1])
    t_alpha.append(model.tvalues[0])
    t_beta.append(model.tvalues[1])
    r_squared.append(model.rsquared)
      


# In[50]:


stats = { "𝛼": alphas,"t-stats for 𝛼" : t_alpha, "𝛽" : betas, "t-stats for 𝛽": t_beta, "R2": r_squared}

stats = pd.DataFrame(stats, index = label)

stats


# In[51]:


# manually checked if beta matches the P1 beta above, it does.

numpy.cov(df["P1"], df["MktRF"])/numpy.var(df["MktRF"], ddof = 1)


# In[52]:


print(stats[stats["t-stats for 𝛼"] > 1.96].index.tolist()) # portfolios with significant alphas 
                                                            # larger than positive critical t value

print(stats[stats["t-stats for 𝛼"] < -1.96].index.tolist()) # portfolios with significant betas
                                                            # smaller than negative critical t value


#total significant

count_significant = stats[stats["t-stats for 𝛼"] > 1.96].shape[0] + stats[stats["t-stats for 𝛼"] < -1.96].shape[0]

print(count_significant)


# In[53]:


stats.to_excel('stats.xlsx')


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


#select excess returns and take their mean along axis 0

means = df[label].mean(axis=0)

m, b = numpy.polyfit(betas, means, 1)

print(m)
print(b)

plt.plot(betas, means, 'bo')
plt.plot(betas, m*numpy.array(betas) + b, '-r')
plt.xlabel("Beta")
plt.ylabel("Excess Return")


# # R code for GRS statistics
# 
# port <- read.csv("port.csv") #please copy this to get the saved file, the file is saved with a command above
# 
# excess_returns <- port[1:32,2:21]
# 
# excess_mkt_returns <- port[1:32,24]
# 
# install.packages("GRS.test")
# library(GRS.test)
# 
# GRS.test(excess_returns,excess_mkt_returns)
# 
# ### $GRS.stat = 1.524746
# 
# ### GRS.pval = 0.238339
# 
# 
# ### Coefficients, t-stats and R2s match the table above

# ## 2. Fama - Macbeth Test

# In[56]:


df_merged #from previous portfolio construction


# In[57]:


# get the mean lagged size of stocks by portfolio by month

df_j = df_merged.groupby(["date", "Portfolio"])["MarketCap"].mean().reset_index().rename(columns={"MarketCap":"MarketCap_Mean"})


# In[58]:


df_j = df_j.sort_values(by=["date", "MarketCap_Mean"])

df_j["MarketCap_Mean"] = df_j["MarketCap_Mean"]/1000 # unit will be in millions

df_j['Log Size'] = numpy.log(df_j['MarketCap_Mean']) # take logarithm

df_j


# In[59]:


sizes = df_j["Log Size"]
sizes = numpy.array(sizes)
sizes = sizes.reshape(20,32)

size = ["Size"+str(i) for i in range(1,33)]

size_data = pd.DataFrame(sizes, columns = size)

size_data


# In[60]:


df_famamacbeth = df_famamacbeth.iloc[:,:20]
df_famamacbeth = df_famamacbeth.divide(100) # convert percentage back to proportion
df_famamacbeth


# In[61]:


df_famamacbeth = df_famamacbeth.transpose()
df_famamacbeth


# In[62]:


month= ["M"+str(i) for i in range(1,33)] 
df_famamacbeth.columns = month
df_famamacbeth


# In[63]:


# portfolio betas

stats["𝛽"]


# In[64]:


df_famamacbeth["Beta"] = stats["𝛽"]
df_famamacbeth


# In[65]:


size_data.index = df_famamacbeth.index


# In[66]:


df_famamacbeth = df_famamacbeth.join(size_data)
df_famamacbeth


# In[67]:


#running cross sectional regressions with portfolio returns, portfolio 𝛽etas and log mean sizes

alphas_mac = []
t_alpha_mac = []
p_alpha_mac = []

b1_mac = []
t_b1_mac = []
p_b1_mac = []

b2_mac = []
t_b2_mac = []
p_b2_mac = []

for i in range(32):
    formula = df_famamacbeth.columns[i] + " ~ " + df_famamacbeth.columns[32] + " + "+ df_famamacbeth.columns[i + 33]
    model = sm.ols(formula, data=df_famamacbeth).fit()
    alphas_mac.append(model.params[0])
    t_alpha_mac.append(model.tvalues[0])
    p_alpha_mac.append(model.pvalues[0])
    b1_mac.append(model.params[1])
    t_b1_mac.append(model.tvalues[1])
    p_b1_mac.append(model.pvalues[1])
    b2_mac.append(model.params[2])
    t_b2_mac.append(model.tvalues[2])
    p_b2_mac.append(model.pvalues[2])


# In[68]:


stats2 = { "𝛼": alphas_mac,"t-stats for 𝛼" : t_alpha_mac, "p-value 𝛼" : p_alpha_mac,
          
          "b1" : b1_mac, "t-stats for b1": t_b1_mac, "p_value b1": p_b1_mac,
          
          "b2": b2_mac, "t-stats for b2": t_b2_mac, "p-value b2": p_b2_mac}

stats2 = pd.DataFrame(stats2, index = df.index)

stats2


# In[69]:


stats2.to_excel("stats2.xlsx")


# In[70]:


#select excess returns and take their mean along axis 0

plt.plot(stats2.index.to_timestamp(), stats2.b1, '-b', label='b1')
plt.plot(stats2.index.to_timestamp(), stats2.b2, '-g', label='b2')
plt.legend(loc='upper right')

plt.xlabel("Time")
plt.ylabel("Coefficients")

plt.tight_layout()


# In[71]:


numpy.mean(alphas_mac)


# In[72]:


numpy.mean(numpy.array(b1_mac))


# In[73]:


numpy.mean(b2_mac)


# In[74]:


scipy.stats.ttest_1samp(alphas_mac, popmean = 0)


# In[75]:


scipy.stats.ttest_1samp(b1_mac, popmean = 0)


# In[76]:


scipy.stats.ttest_1samp(b2_mac, popmean = 0)


# In[77]:


formula = df_famamacbeth.columns[0] + " ~ " + df_famamacbeth.columns[32] + " + "+ df_famamacbeth.columns[33]
model = sm.ols(formula, data=df_famamacbeth).fit()
model.summary()


# ## 3. Alphabet investing

# In[78]:


td = pd.read_csv("ticker_info.csv")
td


# In[79]:


td.dropna(inplace=True)
td


# In[80]:


AA = []
AB = []
BB = []
BA = []

AA_PERM = []
AB_PERM = []
BB_PERM = []
BA_PERM = []

for tckr in td["HTICK"]:
    if len(tckr) > 1:
        if tckr[:2] == "AA":
            AA.append(tckr)
            AA_PERM.append(td[td['HTICK']==tckr]['PERMNO'].values[0])
        if tckr[:2] == "AB":
            AB.append(tckr)
            AB_PERM.append(td[td['HTICK']==tckr]['PERMNO'].values[0])
        if tckr[:2] == "BB":
            BB.append(tckr)
            BB_PERM.append(td[td['HTICK']==tckr]['PERMNO'].values[0])
        if tckr[:2] == "BA":
            BA.append(tckr)
            BA_PERM.append(td[td['HTICK']==tckr]['PERMNO'].values[0])


# In[81]:


print(AA)
print(AB)
print(BB)
print(BA)


# In[82]:


print(AA_PERM)
print(AB_PERM)
print(BB_PERM)
print(BA_PERM)


# In[83]:


df = pd.read_csv("stock_data.csv")  # load stock data

df = df[df.RET!="C"]

df.RET = df.RET.astype("float")

df.date = pd.to_datetime(df.date)

df = df.drop(["SHRCD", "EXCHCD", "CUSIP", "FACPR", "FACSHR", "PRC", "SHROUT"], axis=1) #exclude unnecessary columns

#df[df['PERMNO'].isin(AA_PERM)].groupby("date").mean()


# In[84]:


df2 = df.copy(deep=True)
df3 = df.copy(deep=True)
df4 = df.copy(deep=True)


# In[85]:


# returns of the portfolios

AA_Ret = df[df['PERMNO'].isin(AA_PERM)].groupby("date").mean().RET

AB_Ret = df2[df2['PERMNO'].isin(AB_PERM)].groupby("date").mean().RET

BB_Ret = df3[df3['PERMNO'].isin(BB_PERM)].groupby("date").mean().RET

BA_Ret = df4[df4['PERMNO'].isin(BA_PERM)].groupby("date").mean().RET

merged = pd.concat([AA_Ret, AB_Ret, BB_Ret, BA_Ret], axis=1)

merged.columns = ["AA", "AB", "BB", "BA" ]

merged


# In[86]:


# long - short portfolios

merged["AAAB"] = merged["AA"] - merged["AB"]
merged["AABB"] = merged["AA"] - merged["BB"]
merged["AABA"] = merged["AA"] - merged["BA"]
merged["ABAA"] = merged["AB"] - merged["AA"]
merged["ABBB"] = merged["AB"] - merged["BB"]
merged["ABBA"] = merged["AB"] - merged["BA"]
merged["BBAA"] = merged["BB"] - merged["AA"]
merged["BBAB"] = merged["BB"] - merged["AB"]
merged["BBBA"] = merged["BB"] - merged["BA"]
merged["BAAA"] = merged["BA"] - merged["AA"]


# In[87]:


merged


# In[88]:


# drop initial long only portfolios from dataframe

merged = merged.drop(["AA", "AB", "BB", "BA"], axis = 1)
merged


# In[89]:


#load MOM factor data

MOM = reader.DataReader("F-F_Momentum_Factor", "famafrench", start, end)

MOM = MOM[0]

MOM


# In[90]:


#load Fama French 5 factor data

FF_5 = reader.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start, end)

FF_5 = FF_5[0]

FF_5


# In[91]:


merged.index = FF_5.index


# In[92]:


port_matrix = pd.concat([merged, FF_5, MOM], axis=1) #merge all data


# In[93]:


port_matrix


# In[94]:


# convert the returns to percentage to match fama french data

cols = port_matrix.columns[:10]

for col in cols:
    port_matrix[col] = port_matrix[col]*100
    port_matrix[col] = port_matrix[col] - port_matrix["RF"]
        


# In[95]:


port_matrix


# In[96]:


df = port_matrix.rename(columns={'Mkt-RF': 'MktRF'})

df = df.rename(columns = {df.columns[-1]: 'MOM'})


# In[97]:


#running time-series regressions with 5 factor plus momentum model to estimate alphas and t stats

alphas_5 = []
t_alpha_5 = []
pval_5 = []

for col in cols:
    formula = col + " ~ MktRF + SMB + HML + RMW + CMA + MOM"
    model = sm.ols(formula, data=df).fit()
    alphas_5.append(model.params[0])
    t_alpha_5.append(model.tvalues[0])
    pval_5.append(model.pvalues[0])
 


# In[98]:


#running time-series regressions with CAPM model to estimate alphas and t stats

alphas_capm = []
t_alpha_capm = []
pval_capm = []

for col in cols:
    formula = col + " ~ MktRF"
    model = sm.ols(formula, data=df).fit()
    alphas_capm.append(model.params[0])
    t_alpha_capm.append(model.tvalues[0])
    pval_capm.append(model.pvalues[0])
    


# In[99]:


stats3 = { "CAPM 𝛼": alphas_capm,"CAPM t-stats for 𝛼" : t_alpha_capm, "5 Factor + Mom 𝛼" : alphas_5, "5 Factor + Mom t-stats for 𝛼": t_alpha_5}

stats3 = pd.DataFrame(stats3, index = cols)

stats3


# In[100]:


stats3.to_excel('stats3.xlsx')


# In[101]:


#check if alphas are significant at 0.1

print(numpy.array(pval_5) < 0.1)
print(numpy.array(pval_capm) < 0.1)


# In[102]:


#check if alphas are significant at 0.05

print(numpy.array(pval_5) < 0.05)
print(numpy.array(pval_capm) < 0.05)


# In[103]:


#check if alphas are significant at 0.01

print(numpy.array(pval_5) < 0.01)
print(numpy.array(pval_capm) < 0.01)

