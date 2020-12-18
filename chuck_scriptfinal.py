#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Voici les librairies importées
from bs4 import BeautifulSoup;
import requests;
import time;
import pandas as pd;
import psycopg2;
import numpy as np;
from IPython.display import Image, HTML, display, SVG;
import matplotlib.pyplot as plt;
from matplotlib.pyplot import pie, axis, show;
import numpy as np;
from sqlalchemy import create_engine;
from sqlalchemy.orm import sessionmaker;
from nltk.corpus import stopwords;
from wordcloud import WordCloud, STOPWORDS;
import nltk; 
from nltk.tokenize import word_tokenize;
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from collections import Counter;
import seaborn as sns;
from matplotlib.ticker import FormatStrFormatter;
import config;
import pygal;
from pygal_maps_world.maps import World



# In[3]:


#Pour la connexion à la  base de données PostgreSQL 
conn = psycopg2.connect(database="bdd_cmatto", user=config.user, password=config.password, host='127.0.0.1')
cur=conn.cursor()
print(config.user)


# In[4]:


# Fonction qui permet d'insérer données, l'id et les blagues dans la table chucknorris de la base de données bdd_cmatto (schéma public)
def traiteInfo(idjokes, jokes):
        print("%s: %s" % (idjokes, jokes))
        # Comma-separated dataframe columns
        cur.execute("""INSERT INTO public."chucknorris" VALUES (%s, %s) ON CONFLICT DO NOTHING""", (idjokes, jokes))


# In[5]:


dfjokes=[]
dfvotes=[]
dfnotes=[]
dfid=[]

headers = {'User-Agent' : 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}

def recup_page(page): #Procédure qui traite une page
    print('---', page, '---')
    url= "https://chucknorrisfacts.net/facts.php?page={}".format(page)
    print("Récupération de la page ", url)
    # Extraction du document html
    r=requests.get( url, headers=headers)
    # Récupération de tous les blocks qui contiennetn les infos qui nous interessent
    # Utilisation de soup.select avec selection CSS .select
    soup = BeautifulSoup(r.content,'lxml')
    data_chuck = soup. select("#content > div:nth-of-type(n+2)")# va chercher le sous élement enfant à partir du numéro données, cherche p qui est le 2eme element p
    print("Nombre élements: " ,len(data_chuck))
    #2eme boucle sur les blocks récupérés
    for data in data_chuck:
        #print(data)
        jokes = data.select_one("p") #enlever le format liste sinon faire un select("p")
        if jokes is not None:#si jokes non vide
            idjokes = data.select_one("ul")
            idjokes = idjokes['id'][6:]
            note = data.select_one("span.out5Class")
            nbvote = data.select_one("span.votesClass")  
            #print(note.text, nbvote.text, jokes.text)
            traiteInfo(int(idjokes), jokes.text)
            dfjokes.append(jokes.text)
            dfvotes.append(nbvote.text)
            dfnotes.append(note.text)  
            dfid.append(int(idjokes))
    return len(data_chuck)> 2        

#Boucle qui prend toutes les pages qui contiennent plus de 2 éléments , cette boucle s'arretera s'il y a plus de 999 pages
for p in range (1,999):
    recup = recup_page(p)
    print("Pour la p je récupère :", recup)
    if recup == False:
        break
    


# In[ ]:





# In[6]:


#Dataframe pandas réalisée grâce au web scraping, les blagues, le nombre de vote, les notes et l'id.
dfchuck = pd. DataFrame({"blague":dfjokes, "nbvote":dfvotes, "note":dfnotes, "id": dfid})
dfchuck.head()


# In[7]:


#Réalisation d'un nuage de mots en fonction du contenu des blagues
stop_words = set(STOPWORDS) 
stop_words.add("roundhouse")#Ajout de stopwords qui ne permettaient pas d'avoir un beau nuage de mots
stop_words.add("Norris")
stop_words.add("The")
stop_words.add("When")
stop_words.add("n")
stop_words.add("'")
stop_words.add("t")
stop_words.add("s")
stop_words.add("He")
stop_words.add("got")
stop_words.add("There")
stop_words.add("n't")
stop_words.add("If")
stop_words.add("It")
stop_words.add("gets")
stop_words.add("one'")
stop_words.add("will")
stop_words.add("reason")

print(stop_words) #Liste des stopwords
jokes = dfchuck.blague.str.cat(sep=' ')# Fonction pour séparer le text en mot
tokens = nltk.word_tokenize(jokes)
tagged = nltk.pos_tag(tokens)#Séparation des noms, noms propres, adjectifs, ...
noun=[word for word,pos in tagged if pos == 'NN'] #Selection des noms communs

#Filtre du contenu des blagues sans les stopwords
jokesfilter = [w for w in noun if not w in stop_words] 
jokes_counter = Counter(jokesfilter) #Je compte le nombre de mots sans les stops words
sorted_word_counts = sorted(list(jokes_counter.values()), reverse=True) #Je les trie

#Réalisation du nuage de mots
wordcloud = WordCloud(collocations=False, background_color = "white", max_words=1000, stopwords=stop_words).generate(str(jokesfilter))
plt.imshow(wordcloud)
plt.axis("off")
plt.figure(figsize=(10, 5))
plt.show()


# In[8]:


#Nettoyage conversion en int
dfchuck["note"]=pd.to_numeric(dfchuck["note"])

#Nettoyage j'ai envelé les str "Votes" et converti en int
banned=["Votes"] 
removevotes = lambda x: ' '.join([item for item in x.split() if item not in banned])
dfchuck["nbvote"]=dfchuck["nbvote"].apply(removevotes)
dfchuck["nbvote"]=pd.to_numeric(dfchuck["nbvote"])


# In[9]:


dfchuck['categorie']=dfchuck.blague
dfchuck.head()


# In[10]:


#Création des catégories avec des mots ajoutés manuellement
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('soccer|football|superball|karate|martial|basketball|rugby|dance|salsa|tennis'),"Sport", dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('eats|cereal|burger|pizza|food|fries|cheese|hungry|chicken|fried|cookie|breakfast|dinner|potato|potatoes|Thanksgiving|chocolate|milkshake|butter'),"Food",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('alcohol|drinks|drink|drinking|beer|beers|vodka|rhum|soda|coca|coffee|tea|juice'),"Drinks",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('Naruto|Bruce|Lee|Damme|Trump|Biden|Schwarzenegger|King|The Rock|Phelps|Einstein|Jackie-chan|people|Rambo|Pikachu|Harry Potter|Zombie|zombie|Zombies|zombies|clint eastwood|Clint Eastwood|david guetta|David Guetta|People|people|Pokemon|pokemon|Voldemort|Celebrity|Santa|santa|president'),"People",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('Google|Facebook|Twitter|Instagram|Tik Tok|GAFA|Macintosh|Microsoft|the internet|computer|iPad|touchscreen|pc|gmail|Gmail|google'),"GAFA & the web",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('water|ocean|desert|volcano|mountain|Antarctic|lava|Jupiter|Mars|mars|Moon|moon|Earth|earth|forest|space|sun|planet|NASA|everest|planets|here|galaxy|gravity|earthquake|Black-Hole'),"Earth",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('Jesus|God|Adam|Eve|Bible'),"God",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('Superman|superman|Batman|batman|spider-man|Avengers|Iron-man|superhero|super-hero|Marvel|DC Comics|Heroes|Super Heroes|fire'),"Super-Heroes",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('kick|kicks|kicked|kill|kills|killed|kicking|fight|beats|beat|attack|punch|punches|punched|pepper spray|pushup|Army|army'),"Fight!",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('US|France|french|China|Canada|South-America|England|America'),"Country",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('feels|cries|cry|cried|loves|love|loved|felt|scared|happy|calm|emotions|emotion|excited|fear|jealous|fearless|afraid'),"Emotions",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('cats|cat|kitten|dog|mouse|dogs|Gorilla|rabbits|dolphin|monkey|monkeys|birds|bird|sharks|dinosaurs|dinosaur|grizzly|Grizzly|tigers|tiger'),"Animals",dfchuck.categorie)
dfchuck.categorie =  np.where(pd.Series(dfchuck.categorie).str.contains('Covid-19|Coronavirus|Corona|coronavirus|covid-19|hospital'),"Covid-19",dfchuck.categorie)
dfchuck.loc[dfchuck['categorie'].isin((dfchuck['categorie'].value_counts()[dfchuck['categorie'].value_counts() < 16]).index), 'categorie'] = 'Other'

dfchuck.head(10)
dfchuck.style.set_table_styles([{'selector':'','props':[('border','4px solid #7a7')]}])


# In[11]:


county=dfchuck.groupby(['categorie']).count()['blague'].reset_index() # Creation d'une dataframe avec une colonne categorie et une colonne nombre de blagues par catégorie
county.sort_values('blague',inplace=True)

print(county.columns)

abscissec=county['categorie'].values.tolist()#Conversion des valeurs des colonnes en liste
ordonnéc=county['blague'].values.tolist()

#Pie plot toutes les catégories
# # Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

# # Give color names
from palettable.colorbrewer.qualitative import Set3_12
plt.pie(ordonnéc, labels=None, colors=Set3_12.hex_colors, autopct='%.0f%%', pctdistance=1.2, textprops={'fontsize': 8})
plt.legend(abscissec, loc="center right",bbox_to_anchor=(1.2, 0, 0.5, 1))
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title("Catégorie de blagues les plus populaires\n" + "Chuck Norris Facts")
plt.show()


# In[12]:



#Bar plot toutes les catégories
y_pos = np.arange(len(abscissec))
# Create horizontal bars
plt.barh(y_pos, ordonnéc, color='lightblue')
plt.title("Les catégories les plus populaires", fontsize=10)
# Create names on the y-axis
plt.yticks(y_pos, abscissec)
plt.xlabel("Nombre de blagues", fontsize=8)
# Show graphic
plt.show()


# In[13]:



#Dataframe catégorie  et nombre de blagues par catégorie except Other
countywithout=county.loc[(county.categorie != "Other")]

abscisse=countywithout['categorie'].values.tolist()
ordonné=countywithout['blague'].values.tolist()

#Pie Plot
# # Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')

# # Give color names
from palettable.colorbrewer.qualitative import Set3_12
plt.pie(ordonné, labels=None, colors=Set3_12.hex_colors, autopct='%.0f%%', pctdistance=0.85)
plt.legend(abscisse, loc="center right",bbox_to_anchor=(1.2, 0, 0.5, 1))
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title("Catégorie de blagues les plus populaires\n" + "Chuck Norris Facts")
plt.show()


# In[23]:



#Creation dataframe avec colonne catégorie et moyenne notes et cumul votes par catégorie
graphvn=dfchuck.groupby('categorie').agg({'note':'mean', 'nbvote':'sum'})

sns.set_palette("Set2", 2, .75)
sns.set_context("paper")
sns.set_style({'font.serif': 'Utopia'})
plt.style.use(style="seaborn-pastel")
plt.figure(figsize=(15, 6))
plt.title("Cumul du nombre de votes et notes moyennes \n en fonction des catégories",fontsize=20)
plt.ylabel("Cumul du nombre de vote ", color='steelblue',fontsize=20)

ax = graphvn[['nbvote']].unstack('categorie').plot(kind='bar', use_index=True)
ax.grid(False)
ax.set(xlabel='Catégorie')
ax.set_xticklabels(["Animals  ","Country  ","Drinks  ","Earth  ","Emotions  ","Fight  ","Food  " ,"GAFA & the web             ", "God  ","Other  ","People  ","Sport  ","Super Heroes         "],rotation_mode='anchor',rotation=35,fontsize=12)

ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()],fontsize=15)


ax2 = ax.twinx()
ax2.set(ylim=(3, 4))
ax2.plot(graphvn[['note']].values, linestyle='-', marker='.', linewidth=1.0, color = 'r' )
ax2.set_ylabel('Moyenne des notes', color='r',fontsize=20)
ax2.set_yticklabels(['{:,.2f}'.format(x) for x in ax2.get_yticks()],fontsize=15)



# In[24]:


countries=dfchuck.loc[(dfchuck.categorie == "Country") ] #Dataframe pour voir que la catégorie Country

#Réalisation de la carte ou est représenté l'une des meilleures blagues avec le nom d'un Pays 
worldmap_chart = World()

worldmap_chart.title= " Best worldwide Chuck Norris's jokes "
worldmap_chart.add('CHINE            The Great Wall of China was originally created to keep Chuck Norris out. It failed miserably.', ['cn'], color = '#E8537A')
worldmap_chart.add('FRANCE           Chuck Norris won the Tour De France on a bike with no chain and two flat tires.', ['fr'], color =  '#E87653')
worldmap_chart.add(' USA             America went into the Great Depression when Chuck Norris misplaced his 5 dollars.', ['us'], color = '#E89B53')
worldmap_chart.add(" Canada               Canada didn't get sovereinty by asking for it. They have Chuck Norris negotiate with the British.", ['ca'], color = '#FF00FF')
svg_code = worldmap_chart.render()
SVG(svg_code)

#Pour l'avoir sur en svg en local et voir la carte dynamique
# worldmap_chart.render_to_file('worldblagues.svg')


# In[ ]:





# In[27]:


# Connexion à la bdd 
server = "127.0.0.1" 
BDname="bdd_cmatto"
engine=create_engine('postgresql+psycopg2://cmatto:(cCo\'NR8.\"7xx@127.0.0.1/bdd_cmatto')
Session = sessionmaker(bind=engine)
conpg=engine.connect()

#Création de la table chuck
table_name="chuck_table"
dfchuck.to_sql(table_name, conpg,if_exists="replace", index=False)# cette ligne de code permet de créer la table et de la remplacer si elle existe deja 


# In[28]:


with conpg:
    conpg.execute('ALTER TABLE public."chuck_table" ADD PRIMARY KEY ("id");') #Ajout de la clé primaire sur ma table


# In[29]:


#Web scraping des images
urlimg = "https://www.funnybeing.com/100-funny-selected-chuck-norris-memes/"
html = requests.get(urlimg, headers=headers)
srce= html.content
soup = BeautifulSoup(srce,'lxml')
x=soup.select('img[src^="https://www.funnybeing.com/wp-content/uploads/2017/05"]')
print(x)


# In[30]:



dfimg=[]

for img in x:
    dfimg.append(img['src'])
    
for l in dfimg:
    print(l)

#Création d'une dataframe avec des images
dfmemes= pd. DataFrame({"memes": dfimg})
dfmemes.head()


# In[31]:


#Création de la colonne Content
dfmemes['Content']=dfmemes['memes'].str.split('/').str[7]
dfmemes['Content']=dfmemes['Content'].str[:-12]
dfmemes['Content']=dfmemes["Content"].str.replace("-", " ")
dfmemes['Content']=dfmemes['Content'].str.lower()

dfmemes.head()


# In[32]:


#Ajout de catégories
dfmemes["categorie"] = dfmemes.Content
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('soccer|football|superball|karate|martial|basketball|rugby|dance|salsa|tennis|pushup'),"Sport", dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('eats|cereal|burger|pizza|food|fries|cheese|hungry|chicken|fried|cookie|breakfast|dinner|potato|potatoes|Thanksgiving|chocolate|milkshake|butter'),"Food",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('alcohol|drinks|drink|drinking|beer|beers|vodka|rhum|soda|coca|coffee|tea|juice'),"Drinks",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('Naruto|Bruce|Lee|Damme|Trump|Biden|Schwarzenegger|King|The Rock|Phelps|Einstein|Jackie-chan|people|Rambo|Pikachu|Harry Potter|Zombie|zombie|Zombies|zombies|clint eastwood|Clint Eastwood|david guetta|David Guetta|People|Pokemon|pokemon|Voldemort|Celebrity|Santa|santa|president'),"People",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('Google|Facebook|Twitter|Instagram|Tik Tok|GAFA|Macintosh|Microsoft|the internet|computer|iPad|touchscreen|pc|gmail|Gmail|google'),"GAFA & the web",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('water|ocean|desert|volcano|mountain|Antarctic|lava|Jupiter|Mars|mars|Moon|moon|Earth|earth|forest|space|sun|planet|NASA|everest|planets|here|galaxy|gravity|earthquake|Black-Hole'),"Earth",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('magicians|may force'),"God",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('Superman|superman|Batman|batman|spider-man|Avengers|Iron-man|superhero|super-hero|Marvel|DC Comics|Heroes|Super Heroes|fire|superman'),"Super-Heroes",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('kick|kicks|kicked|kill|kills|killed|kicking|fight|beats|beat|attack|punch|punches|punched|pepper spray|army|Army'),"Fight!",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('US|France|french|China|Canada|South-America|England|America|american'),"Country",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('feels|cries|cry|cried|loves|love|loved|felt|scared|happy|calm|emotions|emotion|excited|fear|jealous|fearless|afraid'),"Emotions",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('cats|cat|kitten|dog|mouse|dogs|Gorilla|rabbits|dolphin|monkey|monkeys|birds|bird|sharks|dinosaurs|dinosaur|grizzly|Grizzly|tigers|tiger'),"Animals",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('Covid-19|Coronavirus|Corona|coronavirus|covid-19|built'),"Covid-19",dfmemes.categorie)
dfmemes.categorie =  np.where(pd.Series(dfmemes.categorie).str.contains('lego'),"Other",dfmemes.categorie)


# In[33]:


searchvalues=['Sport','Drinks','God',"Food",'People','GAFA & the web',"Super-Heroes","Fight!","Country",'Emotions','Animals','Other','Earth', "Covid-19"]

#Dataframe de l'url de image, du "titre de l'image" et de la catégorie
filtredf = dfmemes[dfmemes['categorie'].isin(searchvalues)]

filtredf.head()


# In[34]:


memes_list = filtredf['memes'].tolist()

filtredf['display_img'] = memes_list

filtredf.reset_index()

#Fonction pour avoir les éléments d'un code source 
def path_to_image_html(path):
     return '<img src="'+ path + '" width="60">'

pd.set_option('display.max_colwidth', 1)
#Affichage de l'image dans dataframe
HTML(filtredf.to_html(escape=False,formatters=dict(display_img=path_to_image_html)))


# In[35]:


#Réalisation d'une dataframe qui rassemble les memes, le "titre des images", les catégories, les url des images
filtre= filtredf.drop(filtredf.index[[0,1,5,25,28]])
filtre.groupby('categorie', as_index=False) 

filtre.head()


# In[36]:


#Création de la dataframe , une image = une catégorie
dfimg_categorie=filtre.groupby('categorie',as_index=False).first()
del dfimg_categorie['memes']
print(dfimg_categorie)


# In[37]:


#Connexion à la base de données
server = "127.0.0.1" 
BDname="bdd_cmatto"
engine=create_engine('postgresql+psycopg2://cmatto:(cCo\'NR8.\"7xx@127.0.0.1/bdd_cmatto')
Session = sessionmaker(bind=engine)
conpg=engine.connect()

#Création de la table d'images sur POSTGRESQL
table_name1="image_table"
dfimg_categorie.to_sql(table_name1, conpg,if_exists="replace", index=False)


# In[38]:


with conpg:
    conpg.execute('ALTER TABLE public."image_table" ADD PRIMARY KEY ("categorie");')
    


# In[ ]:




