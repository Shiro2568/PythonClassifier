from sklearn import tree 

t = tree.DecisisionTreeClassifier(criterion='entropy')

t.score(test_attributes,test_labels)

t.predict(example_attributes)
cross_val_score(t,all_attributes,all_label)