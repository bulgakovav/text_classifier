import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
stop_words = stopwords.words("russian")
 

file = r'C:\Users\dno\AppData\Local\Programs\Python\Python36-32\df\news_train.txt'
test_file = r'C:\Users\dno\AppData\Local\Programs\Python\Python36-32\df\news_test.txt'


#к формату csv (rt - чение текста( ос убирает знаки переноса строки) )
#учитель
train_data = list(csv.reader(open(file, 'rt', encoding="utf8"), delimiter='\t'))
rubric = [] #рубрики новостей (выделили из учителя)
article=[]  #содержания новостей
for iter in train_data:
    rubric.append(iter[0])
    article.append(iter[2])

#к формату csv (rt - чение текста( ос убирает знаки переноса строки) )
#данные для классификации
test_data = list(csv.reader(open(test_file, 'rt', encoding="utf8"), delimiter='\t'))
#аналогично: разбиваем на заголовки и содержимое
title = []
test_article = []
for iter in test_data:
    title.append(iter[0])
    test_article.append(iter[1])


#токенизация
#Вес(частота употребления) некоторого слова пропорционален количеству употребления этого слова в документе, и обратно пропорционален частоте употребления слова в других документах коллекции.
vectorizer = TfidfVectorizer(max_features=12350, smooth_idf=True, stop_words=stop_words)
train = vectorizer.fit_transform(article)#(слово - вес)

#наивный Байесовский классификатор (мультинональный)
cls = MultinomialNB() 
#преобразуем в массив
#составляем словарь: слово - рубрика
cls.fit(train, rubric) 



#создаем текстовый файл, куда будут записываться результаты
f=open('results.txt','w')
for iter in test_article: #для каждого новостного текста (новое)
    wIter = vectorizer.transform([iter]) #определяем вес новых слов
    wIter = wIter.toarray()#преобразовали в массив
    res = cls.predict(wIter)#предсказываем категорию нашей новой новости
    answer = str(res) 
    f.write(answer[2:len(answer) - 2]+'\n')#записали в файл с переходом на новую строчку
f.close()

    
