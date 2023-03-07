#I am going to solve this problem in both k-means and hierarchical clustering and check which clustering is good using silhouette_score
##################################### All packages #####################################
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

#taking data
telco=pd.read_excel("C:/Users/yamini/Desktop/Telco_customer_churn/Telco_customer_churn.xlsx")
telco               #7043 rows

#checking null values and duplicated values
telco.isna().sum()
telco.duplicated().sum()
telco.columns

#checking for zero variance varibales
(telco["Count"]==1).all()            
(telco["Quarter"]=="Q3").all()
#count and quarter columns are not needed as there are same values.

################################### Manual dimension reduction ############################
#Instead of using PCA, i am doing dimension reduction manually.
#Customer Id, count, Quarter variables are not needed as they had zero variance in columns and id is not needed.
 
#i can take only number of referrals, instead of taking "referred a friend". So that i get the whole count of referrals.

#'Tenure in Months', 'Offer' is needed as it tells how many months customers used the service and on which offer.

#'Phone Service', 'Multiple Lines' are there then the customers are more interested in service.

#'Avg Monthly Long Distance Charges' are not needed, as it covers in "Total Revenue"

#'Internet Service', 'Internet Type' are needed. But instead of taking the Service, i can just take "internet Type".

#'Avg Monthly GB Download','Online Security', 'Online Backup', 'Device Protection Plan','Premium Tech Support', 'Streaming TV',
#'Streaming Movies', 'Streaming Music', 'Unlimited Data',"contract" are needed.

# 'Paperless Billing' are not needed. It doesnot really evaluate the customers churn.

#'Payment Method' is needed as we can check how people are interested to pay.

#instead of 'Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges'; i can 
#take 'Total Revenue' 
telco.columns

#So for now i have to remove these varaibles:
#'Customer Id', 'count', 'Quarter', 'referred a friend','Avg Monthly Long Distance Charges', 'Monthly Charge', 'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges'
telco1=telco.drop(["Customer ID","Count","Quarter","Referred a Friend",'Internet Service',"Avg Monthly Long Distance Charges",'Paperless Billing',"Monthly Charge","Total Charges","Total Refunds","Total Extra Data Charges","Total Long Distance Charges"],axis=1)
telco1.columns
telco1.head(5)

##################################### doing preprocessing for continuous data #################
telco2=telco1[["Number of Referrals","Tenure in Months","Avg Monthly GB Download","Total Revenue"]]
telco2

#checking for outliers
#Boxplot for number of referrals
plt.boxplot(telco2["Number of Referrals"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco2["Number of Referrals"].mean()-3*telco2["Number of Referrals"].std())
print("right side values: ", telco2["Number of Referrals"].mean()+3*telco2["Number of Referrals"].std())
telco2[(telco2["Number of Referrals"]>10.955465001757492 ) | (telco2["Number of Referrals"]< -7.051730797583136)]
#Referrals can be 11 people, if it was like more, then it would be suspicious. So i dont think these are outliers.So i am not removing

#Boxplot for avg monthly gb download
plt.boxplot(telco2["Avg Monthly GB Download"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco2["Avg Monthly GB Download"].mean()-3*telco2["Avg Monthly GB Download"].std())
print("right side values: ", telco2["Avg Monthly GB Download"].mean()+3*telco2["Avg Monthly GB Download"].std())
telco2[(telco2["Avg Monthly GB Download"]>81.77222654376759 ) | (telco2["Avg Monthly GB Download"]< -40.7414158097054)]
#i dont think there is any outliers in downloading GB variable, as it had 91 rows. i dont think no outliers.

#boxplot for total revenue
plt.boxplot(telco2["Total Revenue"])
plt.title("boxplot")
plt.show()

print("left side values: ", telco2["Total Revenue"].mean()-3*telco2["Total Revenue"].std())
print("right side values: ", telco2["Total Revenue"].mean()+3*telco2["Total Revenue"].std())
telco2[(telco2["Total Revenue"]>11629.992680334672 ) | (telco2["Total Revenue"]< -5561.234568734493)]
telco1.iloc[5491]     #the above all are two year contract. So it is possible to get high total revenue..I dont think no outliers.

#Histograms
plt.hist(telco2["Number of Referrals"])
plt.show()

plt.hist(telco2["Tenure in Months"])
plt.show()

plt.hist(telco2["Avg Monthly GB Download"])
plt.show()

plt.hist(telco2["Total Revenue"])
plt.show()

#normalization
#So all the continuous variables are good to go for normalization
telco_norm=normalize(telco2)
telco_norm

telco_continuous=pd.DataFrame(telco_norm)
telco_continuous

#changing variables names
telco_continuous=telco_continuous.rename(columns={0:"Number of Referrals",1:"Tenure in Months",2:"Avg Monthly GB Download",3:"Total Revenue"})
telco_continuous

##################################### Doing preprocessing for discrete data ##############
#Doing one hot encoding to all the discrete variables at once
enc=OneHotEncoder(handle_unknown="ignore")
enc_var=pd.DataFrame(enc.fit_transform(telco1[["Offer","Phone Service","Multiple Lines","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Contract","Payment Method"]]).toarray())
enc_var.head(5)
enc.get_feature_names_out(["Offer","Phone Service","Multiple Lines","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Contract","Payment Method"])

#changing names for discrete variables
enc_var=enc_var.rename(columns={0:'Offer_None', 1:'Offer_Offer A', 2:'Offer_Offer B', 3:'Offer_Offer C',
       4:'Offer_Offer D', 5:'Offer_Offer E', 6:'Phone Service_No',
       7:'Phone Service_Yes',8:'Multiple Lines_No',9: 'Multiple Lines_Yes',
       10:'Internet Type_Cable',11: 'Internet Type_DSL',
       12:'Internet Type_Fiber Optic', 13:'Internet Type_None',
       14:'Online Security_No',15: 'Online Security_Yes',16: 'Online Backup_No',
       17:'Online Backup_Yes',18: 'Device Protection Plan_No',
       19:'Device Protection Plan_Yes',20: 'Premium Tech Support_No',
       21:'Premium Tech Support_Yes',22: 'Streaming TV_No', 23:'Streaming TV_Yes',
       24:'Streaming Movies_No',25: 'Streaming Movies_Yes',
       26:'Streaming Music_No', 27:'Streaming Music_Yes', 28:'Unlimited Data_No',
       29:'Unlimited Data_Yes', 30:'Contract_Month-to-Month',
       31:'Contract_One Year', 32:'Contract_Two Year',
       33:'Payment Method_Bank Withdrawal',34: 'Payment Method_Credit Card',
       35:'Payment Method_Mailed Check'
                                      })
enc_var

##################################### joining both continuous and discrete data ################
#joining both continuous and discrete data
telco3=telco_continuous.join(enc_var)
telco3.head(2)

####################################### K-means clustering #####################################
#taking kmeans parameters for elbow curve
kmeans_kwargs={"init":"k-means++",
               "max_iter":300,
               "n_init":1,
               "random_state":49
}

sse=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k,**kmeans_kwargs)
    kmeans.fit(telco3)
    sse.append(kmeans.inertia_)
    
#elbow curve
plt.plot(range(1,11),sse)
plt.xticks(range(1,11))
plt.xlabel("number of clusters")
plt.ylabel("sse")
plt.show()

#finding the elbow
k1=KneeLocator(range(1,11),sse,curve="convex",direction="decreasing")
k1.elbow                #4 clusters we have to take according to the elbow curve

#doing kmeans
kmeans=KMeans(init="k-means++", n_init=1, max_iter=300, n_clusters=4, random_state=49)
kmeans.fit(telco3)
kmeans.inertia_
kmeans.n_iter_

#Hyper parameter tuning
kmeans=KMeans(init="k-means++", n_init=1, max_iter=350, n_clusters=4, random_state=58)
kmeans.fit(telco3)
kmeans.inertia_            ###nothing changes much

#kmeans clusters
kmeans.labels_

######################################### Hierarchical clustering ########################
#finding distance
z=linkage(telco3,metric="euclidean", method="complete")

#dendrogram
plt.figure(figsize=(15,8)) 
plt.title("hierarchical clustering dendrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z)
plt.show()

#According to dendrogram, we have to take 4 clusters.
telco_hier=AgglomerativeClustering(n_clusters=4,linkage="complete",affinity="euclidean").fit(telco3)

#hierarchichal clustering labels
telco_hier.labels_

################################### evaluating clusters through silhouette_score  E#####################
kmeans_silhouette = silhouette_score(telco3, kmeans.labels_).round(2)
hier_silhouette = silhouette_score(telco3, telco_hier.labels_).round(2)

kmeans_silhouette
hier_silhouette
#Higher silhouette score is better. 
#both the cluster model are not suited for this as it is not at all nearer to 1. But if we take which 
#once is better, then in this case it is kmeans... 

#taking the kmeans labels
kmeans.labels_

#making labels into a seperate column
clusters=pd.Series(kmeans.labels_)
clusters

#adding clusters varaible to dataframe
telco3["clusters"]=clusters
telco3.head(2)

#taking overall cluster results using mean with groupby clusters
telco_clusters=telco3.iloc[:,0:40].groupby(telco3.clusters).mean().round(4)
telco_clusters

#saving the dataset with clusters
telco3.to_csv("telco_cluster_churn_labels.csv")

##################################### Cluster 1 #############################
#if we check cluster 0, they reffered the service to least number of people and had less tenure of months.
#Their gb download is medium and using offer C and offer D more.
#They had more multiple lines and used fiber optics for internet.
#They had online security and backup with device protection plan.
#They had premium tech support with streaming tv,movies, music.
#They took unlimited data with month-to-month contract.
#They choose bank withdraw payment more.

#Solution: As they are using bank withdraw more, we can offer banks offers and online streaming 
#offers. (Offer C and Offer D)

###################################### cluster 2 ###################################
#cluster 2 is having more number of referrals with more tenure in months.
#Their monthly gb download is very less and they are using offer D.
#They are using more phone service with no or very less multiple lines.
#They are not using any internet. They barely managing the service with no particular contract.
#They are choosing credit card and mailed checks more.

#Solution:As they are using more phone service, we can offer small budget services with out internet.
#We can offer credit card advantages also. (offer D)

##################################### cluster 3 ####################################
#They had less number of referrals and less tenure in months. 
#They are using less monthly gb for download with offer A and offer B.
#They are using very less phone service and using cabel and DSL internet.
#They are having high online security,backup, protection plan and premium tech support.
#They are using unlimited data with streaming services.
#These are the customers who are having 1 year and 2 year plan.
#They are using bank withdraw and credit card more.

#Solution: They are using internet for only streaming services for more months. So we can give offers
#on streaming services and credit card, bank payments. (Offer A and Offer B)

###################################### Cluster 4 ####################################
#They had less referrals and tenure in months.
#Their gb download for month is more and using offer D and offer E. 
#They are using less phone service with cabel and DSL internet.
#they had very less online backup and protectin plan.
#They are using streaming services also less.
#They had unlimited with  month to month contract and using credit and bank withdraw more.

#solution: These customers are using less streaming services, phone service but using more data to
#download. So we can offer data realted offers and credit card offers. (Offer D and Offer E)


####################################### Customer retention solutions ####################
#We should provide services to customers according to offers:
#Offer A: Online Streaming offers, credit card and bank offer
#Offer B: Online Streaming offers, credit card and bank offer
#Offer C: Online streaming offers, credit card and bank offers.
#Offer D: Small package offers, credit card and bank offer
#Offer E: data offers, small package offers, credit card and bank offer
























































    













































































