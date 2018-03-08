import operator
# features and corresponding importance as a list of tuples
feats = sorted([(X.columns[i], j) for (i,j) in enumerate(rf.feature_importances_)], key = operator.itemgetter(1), reverse = True)

#temporary dataframe
df = pd.DataFrame()
df['Features'] = [i for (i,j ) in feats]
df['Importance'] = [j for (i,j) in feats]

#plot
df[:20].plot(kind = 'barh', x = 'Features', figsize = (60,20), fontsize = 25)
plt.savefig('Feature Importances')
plt.show()

# No. of unique Registraion IDs
len(df2['Registration ID'].unique()) 
# Output : 1909 . So there are 21 entries (1930 - 1909) that occure more than once