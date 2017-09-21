import pickle
import numpy as np
import operator
from numpy import linalg as LA

def readFile():

	#number_nodes = 0

	#id_to_title = {}
	#title_to_id = {}

	'''
	#open file
	pagelinks = open("viwiki-20170901-pagelinks.sql",'r')
	pages = open("viwiki-20170901-page.sql",'r')

	#read pages.sql and store object id_to_title and title_to_id
	#count = 0
	print("read file page.sql")
	for line in pages :
		words = line.split(' ')
		if words[0] == 'INSERT' or words[0] == 'insert' :
			values = words[4]
			values = values[1:]
			values = values.split('),(')
			for value in values :
				#count = count + 1
				attrs = value.split(',')
				ide = int(attrs[0])
				title = attrs[1]+'_'+attrs[2]
				id_to_title[ide] = title
				title_to_id[title] = ide
	'''

	#save to file
	#file1 = open('id_to_title.pkl','rb')
	#file2 = open('title_to_id.pkl','rb')
	#id_to_title = pickle.load(file1)
	#title_to_id = pickle.load(file2)

	#file1.close()
	#file2.close()
	#number_nodes = count
	#print('number_nodes = '+str(number_nodes))

	#in-degree of each page
	in_degree = {}
	#list point to page i
	link_from = {}

	#read pagelinks.sql and build graph
	'''print("read file pagelinks.sql")
	for line in pagelinks:
		words = line.split(' ')		
		if words[0] == 'INSERT' or words[0] == 'insert' :
			values = words[4]
			values = values[1:]
			values = values.split('),(')
			for value in values :
				attrs = value.split(',')
				id_from = int(attrs[0])
				title = attrs[1]+'_'+attrs[2]
				if not title in title_to_id:
					continue
			
				id_to = title_to_id[title]
				if id_to in link_from :
					link_from[id_to].append(id_from)
				else :
					link_from[id_to] = []

				if id_to in in_degree :
					in_degree[id_to] += 1
				else :
					in_degree[id_to] = 1
	'''
	file3 = open('in_degree.pkl','rb')
	file4 = open('link_from.pkl','rb')
	in_degree = pickle.load(file3)
	link_from = pickle.load(file4)

	#filescreen.write('------------in_degree---------------------\n')
	#filescreen.write(str(in_degree) + '\n')
	#filescreen.write('------------link_from---------------------\n')
	#filescreen.write(str(link_from) + '\n')


	file3.close()
	file4.close()

	print("Build Graph")
	#get top k most prestige page
	k = 15000
	topKpage = dict(sorted(in_degree.iteritems(),key=operator.itemgetter(1), reverse=True)[:k])

	graph = np.zeros((k,k))

	new_id_2id_page = {}
	id_page_2new_id = {}
	it = 0	
	for id_page in topKpage :
		new_id_2id_page[it] = id_page
		id_page_2new_id[id_page] = it
		it += 1

	for id_page in topKpage :
		id_pages_from = link_from[id_page]
		for id_page_from in id_pages_from:
			if id_page_from in topKpage:
				id_to = id_page_2new_id[id_page]
				id_from = id_page_2new_id[id_page_from]
				graph[id_from][id_to] = 1.0

	#print(str(graph)+'\n')
	#normalize graph
	v = np.sum(graph,axis=1)

	i_v = 0
	for v_v in v:
		if v_v == 0:
			graph[i_v] = np.full((1,k),1.0/k)
			v[i_v] = 1.0
		i_v += 1

	#print(str(graph)+'\n')
	graph = graph/v[:,None]
	#print(str(graph))

	return graph, new_id_2id_page
	#return new_id_2id_page, id_to_title

def rank(M, d, ep) :
	print("Compute rank")
	N = 15000
	v = np.random.rand(N,1)
	v = v / LA.norm(v,1)
	last_v = np.ones((N,1))*100
	M_hat = (d*M) + ((1-d)/N)*np.ones((N,N))

	while(LA.norm((v-last_v),2) > ep):
		print("v-last_v = "+str(LA.norm((v-last_v),2)))		
		last_v = v
		v = M_hat.dot(v);
	
	#np.save('rank.npy',v)
	return v

def outputResult(v, id_pages, titles):
	print("Write Result")	
	f = open('Assignment01_20134501_PhanAnhTu.txt','w')
	f.write('Rank \t \t Id \t \t Title \n')
	k = 1000
	
	topKrank = sorted(enumerate(v), key=lambda x:x[1], reverse=True)[:k]
	for page in topKrank :
		id_page = page[0]
		rank = page[1]
		idx = id_pages[id_page]
		title = titles[idx]
		f.write(str(rank) + '\t \t' + str(idx) +'\t \t'+ str(title) +'\n')

	f.close()		
	

M, id_pages = readFile()

#file2 = open('titles.pkl','rb')
#pickle.dump(id_pages, file1, pickle.HIGHEST_PROTOCOL)
#pickle.dump(titles, file2, pickle.HIGHEST_PROTOCOL)
#print("Load id_pages")
#id_pages = pickle.load(file1)
#print("Load titles")
#titles = pickle.load(file2)
#file1.close()
#file2.close()

v = rank(M, 0.85, 0.00001)

#file1 = open('id_pages.pkl','rb')
file1 = open('id_to_title.pkl','rb')

#print("Load v")
#v = np.load('rank.npy')
#id_pages = pickle.load(file1)
titles = pickle.load(file1)
outputResult(v, id_pages, titles)
