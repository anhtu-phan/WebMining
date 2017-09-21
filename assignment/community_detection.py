import snap

def build_graph():

    id_to_title = {}
    title_to_id = {}

    #open file
    pagelinks = open("./community_detection/viwiki-20170420-pagelinks.sql",'r')
    pages = open("./community_detection/viwiki-20170420-page.sql",'r')

    #read pages.sql and store object id_to_title and title_to_id
    print("read file page.sql ....")
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

    G = snap.TUNGraph.New()

    # read pagelinks.sql and build graph
    print("read file pagelinks.sql ....")
    for line in pagelinks:
        words = line.split(' ')
        if words[0] == 'INSERT' or words[0] == 'insert' :
            values = words[4]
            values = values[1:]
            values = values.split('),(')
            for value in values :
                attrs = value.split(',')
                title = attrs[1] + '_' + attrs[2]

                if not title in title_to_id:
                    continue

                id_from = int(attrs[0])
                if not G.IsNode(id_from):
                    G.AddNode(id_from)

                id_to = title_to_id[title]
                if not G.IsNode(id_to):
                    G.AddNode(id_to)

                if not G.IsEdge(id_from, id_to):
                    G.AddEdge(id_from,id_to)


    return G, id_to_title

def detect_community(G, id_to_title):

    print('dectect community ....')
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(G, CmtyV)

    f = open('./community_detection/assignment2_Nhom1_TuToanChien.txt','w')

    i = 0
    for Cmty in CmtyV :
        if i == 100:
            break

        f.write('Community '+str(i)+': \n')

        j = 0
        for NI in Cmty:
            if j == 10 :
                break

            title = id_to_title[NI]
            f.write(str(NI)+'\t \t'+str(title)+'\n')
            j += 1

        i+= 1

    f.close()

G, id_to_title = build_graph()
detect_community(G, id_to_title)
