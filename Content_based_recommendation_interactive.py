#report recommender
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.metrics.pairwise import cosine_similarity

#Read the data
df_user_report = pd.read_csv(r'C:\Sid Data\BITS\4th Sem\Sidd code\Content based Recomendation\Users_report__details_crosstab.csv')
#Rename the columns
df_user_report.columns = ["Name", "SSO", "Biz", "Band", "Country", "Product", "Report_Name", "Rating" ]
pivot_on_product = pd.pivot_table(df_user_report, index = 'Product',columns = 'Biz')
print (pivot_on_product)

#Retain only the FDS, FNOBIEE & TABLEAU products
core_product = ['FDS', 'FNOBIEE', 'FNTABLEAU']
fd_user_report_core = df_user_report[df_user_report.Product.isin(core_product)]



#cleanse the data by remocing any report name with the word PROMPT as these are intermediate steps
fd_user_report_core_clean = pd.DataFrame(columns = ["Name", "SSO", "Biz", "Band", "Country", "Product", "Report_Name", "Rating" ])
print(fd_user_report_core_clean.describe())

# Delete all rows of the data frame in order to populate the data
for i, each_row in fd_user_report_core_clean.iterrows():
    fd_user_report_core_clean.drop(index = i, axis = 0, inplace = True)
    
print(fd_user_report_core_clean.describe())
 
# ------- Remove reports with the word PROMPT or TEST-----------------------
for index, row in fd_user_report_core.iterrows():
  s =row['Report_Name']
  #print ('Prompt' in s or 'test' in s, s)
  if (('Prompt' in s or 'test' in s)):
      #print ('skiping recrd with name -->' + s)
      continue
  else:
      fd_user_report_core_clean = fd_user_report_core_clean.append(fd_user_report_core.loc[index], ignore_index=True)
      #print (row)

#cross check the number of rows removed through the clean up
print(fd_user_report_core.describe())
print(fd_user_report_core_clean.describe())
      
#-------------------Process 1----Report to report similarity-----
#Merge the 2 columns Product & Report_Name
fd_user_report_core_clean['Product_Report'] = fd_user_report_core_clean['Product'] + ' ' + fd_user_report_core_clean['Report_Name']   

#Merge the 2 columns Product & Report_Name
fd_user_report_core_prod_report = pd.DataFrame()
fd_user_report_core_prod_report['Product_Report'] = fd_user_report_core_clean['Product_Report']

#remove duplicates from the data drame and retain only unique values
fd_user_report_core_prod_report_no_dup = fd_user_report_core_prod_report.drop_duplicates(subset = 'Product_Report', keep = 'first', inplace = False )
fd_user_report_core_prod_report_no_dup.reindex

#---------------------Bag of words model---------------------------------------------------
# Use the cosine similarity
cv = CountVectorizer( stop_words = 'english', )
fd_report_vector = cv.fit_transform(fd_user_report_core_prod_report_no_dup['Product_Report'])
print (fd_report_vector.toarray())

#Get the data for the vocabulary 
print (cv.vocabulary_)


#----------------------Cosine Similarity-----------------
report_cosine_similar = cosine_similarity(fd_report_vector)
print (report_cosine_similar)
#-----------------------------------------------------------------------
# How to get a index for a matching moie
print (fd_user_report_core_prod_report_no_dup.loc[fd_user_report_core_prod_report_no_dup.Product_Report =='FNOBIEE C&R Undis'].index.values[0])
# How to get a movie for a index
#print (fd_user_report_core_prod_report_no_dup.loc[fd_user_report_core_prod_report_no_dup.index ==4].Product_Report.values[0])
#---------------- Get an index for an report -------------------------
def get_index_by_report_name (report_name):
    return (fd_user_report_core_prod_report_no_dup.loc[fd_user_report_core_prod_report_no_dup.Product_Report == report_name].index.values[0])
# -----------------------------Get a report for an index---------
def get_report_by_index_name (index_name):
    return (fd_user_report_core_prod_report_no_dup.loc[fd_user_report_core_prod_report_no_dup.index == index_name].Product_Report.values[0])

#print (fd_user_report_core_prod_report_no_dup.loc[fd_user_report_core_prod_report_no_dup.index == 1163].Product_Report.values[0])


# get report index for look up
lookup_report = input('Enter the report name for wich similar report look up is needed--->')
try:
    lookup_report_index = get_index_by_report_name(lookup_report)
except IndexError:
    print('Report Name not found Please TRY again')
else:
    print (lookup_report_index)

# Using the index lookup the report_cosine_similar
#similar_reports_are = list(enumerate(report_cosine_similar[5]))
#print (similar_reports_are)
similar_reports_are = list(enumerate(report_cosine_similar[lookup_report_index]))
print (similar_reports_are)

# As we have enumeratedm the values are in the form is (1, X), ( 2, Y), so we define a function to retun the 2nd element 
def return_second_element(x):
    return (x[1])
# Sort the reports based on the cosine similarity which is the second element
sorted_similar_reports_all = sorted(similar_reports_are, key=return_second_element, reverse = True)

#take the second element as the top similairty will be its itself
sorted_similar_reports_ignoreself = sorted_similar_reports_all[1:]
a = sorted_similar_reports_ignoreself[0][1]

#print top 5 recomended report
print ('--------------------- The top 5 reports of ' + lookup_report + '--------------------')
for  count in range(5):
    # Get report by the index
    index = sorted_similar_reports_ignoreself[count][0]
    #get the report name from index
    print ('The ' + str(count) + 'number matching report is  -->' + get_report_by_index_name(index))
    
    
    
    








            




