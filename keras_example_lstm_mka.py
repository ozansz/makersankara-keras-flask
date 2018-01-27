'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

text = ""

with open('mka_quotes.txt', mode='r') as file:
    text = file.read()

text = """Ahmaklar, memleketi Amerikan mandasına, İngiliz himayesine terk etmekle kurtulacak sanıyorlar. Kendi rahatlarını temin etmek için bir vatanı ve tarih boyunca devam edip gelen Türk istiklalini feda ediyorlar! Akıl ve mantığın halledemeyeceği mesele yoktur. Amerika, Avrupa ve bütün uygarlık dünyası bilmelidir ki Türkiye halkı her uygar ve kabiliyetli millet gibi kayıtsız şartsız hür ve müstakil yaşamaya kesin karar vermiştir. Bu haklı kararı bozmaya yönelik her kuvvet, Türkiye'nin ebedi düşmanı kalır. Anadolu, en büyük hazinedir. Artık Türkiye, din ve şeriat oyunlarına sahne olmaktan çok yüksektir. Bu gibi oyuncular varsa, kendilerine başka taraflarda sahne arasınlar. Asla şüphem yoktur ki, Türklüğün unutulmuş medeni vasfı ve büyük medeni kabiliyeti, bundan sonraki inkişafı ile âtinin yüksek medeniyet ufkunda yeni bir güneş gibi doğacaktır. Bu söylediklerim hakikat olduğu gün, senden ve bütün medeni beşeriyetten dileğim şudur: Beni hatırlayınız. Az zamanda çok ve büyük işler yaptık. Bu işlerin en büyüğü, temeli, Türk kahramanlığı ve yüksek Türk kültürü olan Türkiye Cumhuriyeti'dir. Ben, 1919 yılı mayısı içinde Samsun'a çıktığım gün elimde maddi hiçbir kuvvet yoktu. Yalnız büyük Türk milletinin soyluluğundan doğan ve benim vicdanımı dolduran yüksek ve manevi bir kuvvet vardı. İşte ben bu ulusal kuvvete, bu Türk milletine güvenerek işe başladım. Ben manevî miras olarak hiçbir nas-ı katı, hiçbir dogma, hiçbir donmuş, kalıplaşmış kural bırakmıyorum. Benim manevî mirasım ilim ve akıldır. Benden sonra beni benimsemek isteyenler, bu temel mihver üzerinde akıl ve ilmin rehberliğini kabul ederlerse manevî mirasçılarım olurlar. Beni görmek demek mutlaka yüzümü görmek değildir. Benim fikirlerimi, benim duygularımı anlıyorsanız ve hissediyorsanız bu kafidir. Benim naçiz vücudum bir gün elbet toprak olacaktır fakat Türkiye Cumhuriyeti sonsuza dek yaşayacaktır. Ve Türk milleti güven ve mutluluğun kefili olan ilkelerle, uygarlık yolunda, tereddütsüz yürümeye devam edecektir. Benim Türk milletine, Türk cemiyetine, Türklüğün istikbaline ait ödevlerim bitmemiştir, siz onları tamamlayacaksınız. Siz de sizden sonrakilere benim sözümü tekrar ediniz. Bir hükûmet iyi midir, fena mıdır? Hangi hükümetin iyi veya fena olduğunu anlamak için, "Hükümetten gaye nedir?" bunu düşünmek lazımdır. Hükûmetin iki hedefi vardır. Biri milletin korunması, ikincisi milletin refahını temin etmek. Bu iki şeyi temin eden hükûmet iyi, edemeyen fenadır. Bir kelime ile ifade etmek gerekirse, diyebiliriz ki yeni Türkiye Devleti bir halk devletidir; halkın devletidir. Mazi kurumları ise bir şahıs devleti idi, şahıslar devleti idi. Bir toplumun eksikliği ne olabilir? Ulusu ulus yapan, ilerleten ve geliştiren güçler vardır: Düşünce güçleri, sosyal güçler. Düşünceler, anlamsız, yararsız, akla sığmaz saçmalarla dolu olursa o düşünceler hastalıklıdır. Bir de toplumsal yaşayış, akıldan mantıktan uzak, yararsız, zararlı birtakım görenek ve geleneklerle dopdolu olursa yaşama sayılamaz. İlerleyemez, gelişemez, inmeliler gibi olduğu yerde bocalar kalır. Birbirimize daima gerçeği söyleyeceğiz. Felaket ve saadet getirsin, iyi ve fena olsun, daima gerçekten ayrılmayacağız. Biz cahil dediğimiz zaman, mektepte okumamış olanları kastetmiyoruz. Kastettiğimiz ilim, hakikati bilmektir. Biz, her vasıtadan yalnız ve ancak bir tek temel görüşe dayanarak yararlanırız. O görüş şudur: Türk milletini medenî dünyada lâyık olduğu mevkie yükseltmek, Türkiye Cumhuriyeti’ni sarsılmaz temelleri üzerinde her gün daha çok güçlendirmek... ve bunun için de istibdat fikrini öldürmek... Bizce, Türkiye Cumhuriyeti anlamınca kadın, bütün Türk tarihinde olduğu gibi bugün de en saygın düzeyde, her şeyin üstünde yüksek ve şerefli bir varlıktır. Bizi yanlış yola sevk eden kötü yaradılışlılar, çok kere din perdesine bürünmüşler, saf ve temiz halkımızı hep din kuralları sözleriyle aldatagelmişlerdir. Tarihimizi okuyunuz, dinleyiniz... Görürsünüz ki milleti mahveden, esir eden, harap eden fenalıklar hep din örtüsü altındaki küfür ve kötülükten gelmiştir. Bizim barış ülküsüne ne kadar bağlı olduğumuzu, bu ülkünün güvenlik altına alınmasındaki dileğimizin ne kadar esaslı bulunduğunu izaha lüzum görmüyorum. Bizim milletimiz esasen demokrattır. Kültürünün, geleneklerinin en derin maziye ait evreleri bunu doğrular. Bugünkü hükümetimiz, devlet örgütümüz doğrudan doğruya milletin kendi kendine, kendiliğinden yaptığı bir devlet örgütü ve hükumettir ki onun ismi Cumhuriyettir. Artık hükumet ile millet arasında mazideki ayrılık kalmamıştır. Hükümet millettir ve millet hükümettir. Artık hükümet ve hükümet mensupları kendilerinin milletten ayrı olmadıklarını ve milletin efendi olduğunu tamamen anlamışlardır."""

#with open('rte_konusma.txt', 'r') as rte:
#    for line in rte:
#        text += line.lower()
#
#print('='*30)
#print(text)
#print('='*30, end='\n\n\n')

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    #start_index = random.randint(0, len(text.split(" ")) - 9)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        #sentence = " ".join(text.split(" ")[start_index:start_index+15])
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            #x_pred = np.zeros((1, 8, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
