from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
import random
import csv
import datetime

chatbotname="Chatbot: " # Defining chatbot name
def dict_load(): #Loading Q&A sample dataset and creation of small talk dictionary

   ques_n_ans_set = {"what are arizona's symbols": "The newest adopted symbol of Arizona is the Colt Single Action Army in 2011.,List of Arizona state symbols",
   "what are arizona":"Arizona became the second state to adopt after Utah adopted the Browning M1911", 
"what is adoration catholic church":"Eucharistic adoration is a practice in the Roman Catholic Church , and in a few Anglican and Lutheran churches, in which the Blessed Sacrament is exposed and adored by the faithful.", 
"when the body is systemic":"Systemic refers to something that is spread throughout, system-wide, affecting a group or system such as a body, economy, market or society as a whole.",
"what are SLR cameras":"A single-lens reflex (SLR) camera is a camera that typically uses a mirror and prism system (hence ""reflex"", from the mirror's reflection) that permits the photographer to view through the lens and see exactly what will be captured, contrary to viewfinder cameras where the image could be significantly different from what will be captured.",
"what are the quad muscles":"It is the great extensor muscle of the knee, forming a large fleshy mass which covers the front and sides of the femur .",
"when did the titanic sink":"RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean on 15 April 1912 after colliding with an iceberg during her maiden voyage from Southampton , UK to New York City , US.",
"what is considered a large car":"A full-size car is a marketing term used in North America for an automobile larger than a mid-size car"}

   dict_sm_talk_greet = {
   'hey' : 'Hey!',
   'hi': 'Hi!',
   'hello': 'Hello!',
   'yo': 'yo yo!!',
   'a joke': 'random_joke',
   'how are you': 'I am fine. Thankyou. greeting',
   'how are you doing': 'I am doing good. Thankyou. greeting',
   'how do you do': 'I am great. Thanks. greeting',
   'how are things': 'Going good. Thankyou. greeting',
   'whats up': 'Just chilling ',
   'thank': 'You are welcome ',
   'thankyou': 'You are welcome ',
   'good morning': 'Good Morning greeting',
   'good afternoon': 'Good Afternoon greeting',
   'good evening': 'Good Evening greeting',
   'good day': 'Wish you the same',
   'Have a great day': 'Wish you the same',
   'who are you': 'I am chatbot',
   'your name': 'I am chatbot',
   'what are you doing': 'Talking to you username'
   }
   return ques_n_ans_set, dict_sm_talk_greet

def tfidf_cosine_similarity_func(data_collection, query, stopwords=[]): # Cosine-Similarity calculation
   query = [query]
   sb_stemmer = SnowballStemmer('english')
   data_collection_stem = [sb_stemmer.stem(word) for word in data_collection] 
   query_stem = [sb_stemmer.stem(word) for word in query]
   tfidf = TfidfVectorizer(use_idf=True, sublinear_tf=True, stop_words=stopwords)
   tfidf_data_collection = tfidf.fit_transform(data_collection_stem)
   tfidf_query = tfidf.transform(query_stem)
   CosSimlarity = cosine_similarity(tfidf_data_collection,tfidf_query).flatten()
   data_collection_index = CosSimlarity.argsort()[-1]
   return CosSimlarity[data_collection_index], data_collection_index


def random_joke_func(): # Random joke generator function
   joke_list=["Why did the chatbot go to the doctor? Because it had a virus!","What do computers eat when they get hungry? Chips!!", "why cricket stadiums are so cool? Because every seat has a fan in it","what is most shocking city in the world? Electricity", "what did cow say when it wanted to watch film? Lets go the moovies", "Did you know that all clouds have dandruff? Thats where snowflakes come from"]
   return(joke_list[random.randrange(0, len(joke_list))])

def name_identifier(name_query): # User name capture function
   name_sentence_list =["my name is", "call me", "name is"]
   similarity,index=tfidf_cosine_similarity_func(name_sentence_list,name_query)
   if similarity > 0.5:
         name = name_query.replace(name_sentence_list[index]," ").strip()
   else:
         name = name_query
   return name.capitalize()


def small_analysis(query, dict_sm_talk_greet, username): # Small talk analysis function
   result=None
   similarity_factor,index=tfidf_cosine_similarity_func(dict_sm_talk_greet,query)
   if similarity_factor > 0.8:
      result=list(dict_sm_talk_greet.values())[index]
      if result.strip()=="random_joke":
         result=random_joke_func()
      elif result.strip().split(" ")[-1]=="username":
         result=result.replace("username",username)
   return result,similarity_factor

def ques_and_answer_analysis(query,ques_n_ans_set): # Qustion-Answer Retrieval function
   result=None
   similarity_factor,index=tfidf_cosine_similarity_func(ques_n_ans_set,query)
   if similarity_factor > 0.5:
      result=list(ques_n_ans_set.values())[index]
      result= "\nHere is what I found from my database: \n" +"\"" +result +"\""+ "\n\nHope this was helpful. :)"
   return result,similarity_factor   


def sentiment_analysis_classifier(query): # Logistic Regression classifier for sentiment analysis
   positive_list=["I am feeling good", "I am good, Thankyou", "I am good", "Im good","I am fine", 
   "doing good", "going good", "I am well"]
   negative_list=["I am feeling sick", "I am sick","I am not well", "Under the weather", "im not good", 
   "not so good","hectic", "not that great","feeling bad" ]
   label_dir = {
   "positive": positive_list, "negative": negative_list
   }
   data = [] 
   labels = []
   for label in label_dir.keys():
      for x in label_dir[label]:
               data.append(x)
               labels.append(label)
   tfidf=TfidfVectorizer(use_idf=True, sublinear_tf=True)
   tfidf_data = tfidf.fit_transform(data)
   tfidf_query = tfidf.transform([query])
   clf = LogisticRegression(random_state=0).fit(tfidf_data, labels)
   predicted = clf.predict(tfidf_query)
   return predicted

def sentiment_analysis_func(final_result, username): # Sentiment Analysis intent calculator
      new_result=final_result.replace("greeting", "How are you "+username+" ?\n\n")
      query = input("\n"+chatbotname+new_result)
      senti=sentiment_analysis_classifier(query)
      if senti == "positive":
         print("\n"+chatbotname+"Looks like you had a great day. Keep Going!!")
      elif senti == "negative":
         query1=input("\n"+chatbotname+"I can feel that you are sad. Do you want me to tell you a joke ?\n")
         if query1 in ["yes","yes please","y","sure"]:
            result=random_joke_func()
            print("\n"+chatbotname+"Sure..")
            print(result)
         else:
            print("\n"+chatbotname+"Sure. No worries. I hope you feel better soon\n")

def calc_input(): # Arithmetic calculator operands input function
   list1=input("\n"+chatbotname+"Enter numbers seperated by spaces\n").strip().split(" ")
   list2 = []
   for i in list1:
      if i:
         list2.append(float(i))
   return list2

def calculator_func(): # Arithmentic calculator operation menu function
   query1=input("\n"+chatbotname+"Do you want to do mathematical calculation?\n")
   if query1 in ["yes","yes please","y","sure"]:
      print("\n"+chatbotname+"sure. Lets do it!\n")
      k=True
      while k is True:
         print("\n"+chatbotname+"I support the following basic mathematical calculations. 1-add,2-subtract,3-multiply,4-Divide,5-Quit&Return\n")
         maths_select=input(""+chatbotname+"Please input your preference.\n").strip()
         if str(maths_select) in ["5", "quit", "Quit", "return", "q", "Quit&Return", "return"]:
            k=False
         else:        
            result = "Sorry. Looks like I couldn't answer that. Please try again\n"
            if str(maths_select) in ["1", "addition", "add"]:
               list1=calc_input()
               if list1:
                  result = 0
                  for x in list1:
                     result = result + x
            elif str(maths_select) in ["2", "subtraction", "subtract"]:
               list1=calc_input()
               if list1:
                  result = list1[0]
                  for x in range (1,len(list1)):
                     result = result - list1[x] 
            elif str(maths_select) in ["3", "multiplication", "multiply"]:
               list1=calc_input()
               if list1:
                  result = 1
                  for x in list1:
                     result = result * x
            elif str(maths_select) in ["4", "division", "divide"]:
               list1=calc_input()
               if list1:
                  result = list1[0]
                  for x in range (1,len(list1)):
                     if list1[x]==0.0:
                        result = "Invalid input. Cannot divide by 0. Please try again"
                        break
                     result = result / list1[x]
            else:
               result = "Sorry. Looks like I couldn't answer that. Please try again"
            print("\n")
            print(chatbotname+str(result))    

def name_retriever(query,username): # User name Retrieval function
   result=None
   query_pattern_list =["who am i", "what my name", "my name"]
   similarity,index=tfidf_cosine_similarity_func(query_pattern_list,query)
   if similarity > 0.5:
      result="You are "+username
   return result,similarity  

def time_calcul_func(query): # Current time function
   result=None
   query_pattern_list =["time", "what current time", "what time now", "current time"]
   similarity,index=tfidf_cosine_similarity_func(query_pattern_list,query)
   if similarity > 0.6:
      result=datetime.datetime.now().strftime('%I:%M:%S %p')
   return result,similarity  

def date_calcul_func(query): # Current date fucntion
   result=None
   query_pattern_list =["date", "today's date", "day today"]
   similarity,index=tfidf_cosine_similarity_func(query_pattern_list,query)
   if similarity > 0.5:
      result=datetime.datetime.today().strftime('%Y-%m-%d')
   return result,similarity  

def calculator_intent_detect_func(query): # Mathematical calculation intent detection fucntion
   result=None
   query_pattern_list =["addition of numbers", "add numbers", "subtraction of numbers", 
   "subtract", "division", "divide", "multiply", "multiplication", "calculate", 
   "calculation of numbers"]
   similarity,index=tfidf_cosine_similarity_func(query_pattern_list,query)
   if similarity:
      result="mathematical_calculation"
   else:
      for x in query:
         if x in ["+", "-", "/", "*", "="]:
            result="mathematical_calculation"
            similarity=1.0               
   return result,similarity  

def intent_analysis(query,username, ques_n_ans_set, dict_sm_talk_greet): # Main Intent detection and analysis fucntion
   intent_dict={-1:"I am sorry!! I cannot help you with this one. Please try again."}
   name_retriever_result,sim_fact_nr=name_retriever(query,username)
   if name_retriever_result:
      intent_dict[sim_fact_nr] = name_retriever_result
   small_talk_result,sim_fact_st=small_analysis(query, dict_sm_talk_greet, username)
   if small_talk_result:
      intent_dict[sim_fact_st] = small_talk_result
   qaa_result,sim_fact_qaa=ques_and_answer_analysis(query, ques_n_ans_set)
   if qaa_result:
      intent_dict[sim_fact_qaa] = qaa_result
   cal_result,sim_cal=calculator_intent_detect_func(query)
   if cal_result:
         intent_dict[sim_cal] = str(cal_result)
   time_result,simtime=time_calcul_func(query)
   if time_result:
      intent_dict[simtime] = str(time_result)
   date_result,simdate=date_calcul_func(query)
   if date_result:
      intent_dict[simdate] = str(date_result)
   final_result = intent_dict[max(intent_dict.keys())]
   if final_result.strip().split(" ")[-1]=="greeting":
      sentiment_analysis_func(final_result, username)
   elif final_result.strip().split(" ")[-1]=="mathematical_calculation":
      calculator_func()
   else:
      print("\n")
      print(chatbotname+str(final_result))



def main(): #Main loop and query input
   ques_n_ans_set, dict_sm_talk_greet = dict_load()
   name_query = input ("\n"+chatbotname+"Hi Iam Chatbot. what is your name?\n")
   name_query = name_query.lower()
   username = name_identifier(name_query)
   print("\n"+chatbotname+"Hi "+username+", How can I help you?")
   stop=True
   while(stop==True):
      input_query = input("\n")
      input_query = input_query.lower().strip("!@#$%^&*()<>,;?")
      if input_query in ['quit', 'q', 'bye', 'exit']:
         stop=False
         print("\n"+chatbotname+"Bye "+username+" Take care!")
      
      else: 
         intent_analysis(input_query, username, ques_n_ans_set, dict_sm_talk_greet)

if __name__ == '__main__':
    main()

