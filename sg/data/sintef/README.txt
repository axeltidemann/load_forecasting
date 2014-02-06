Fra SINTEF Energi v/Nicolai Feilberg har vi fått to ladninger med data:

  1. Et sett med tekstfiler med anonymiserte lastmålinger. Dette er både
      "mandags-" og "onsdagsdata".  Onsdagsdata er mandagsdata med visse
      endringer. Manuell sammenligning av tilfeldig utvalgte filer (emacs
      ediff) tyder på at endringene inkluderer fiksing av sommertid og en del
      små systematiske justeringer (f.eks. 5% endring i en gitt uke).

  2. En ladning med GS2-filer og ymse andre måledata, mottatt 3. februar
      2011. Opprinnelig fikk vi ikke-anonymiserte data, disse ble anonymisert
      som beskrevet under. Backup er tilgjengelig i kryptert format.
==================================================



ANONYMISERING AV GS2-DATA (skrevet mens kode og data lå i samme mappe):
--------------------------------------------------

* GS2-filene har filending .exp eller .gs2. Liste over alle GS2-filene ligger i
  gs2.txt, denne kan genereres med:
$ ./make-list-of-gs2-files.sh

* Ikke-GS2-filer ligger i not_gs2.txt, generert med:
$ find . -not -iname '*.gs2' -and -not -iname '*.exp' -and -not -type d >../not_gs2.txt

* De fleste ikke-GS2-filene har blitt slettet (Buskerud/Statuskoder.txt ble
  beholdt). NB! NB! Merk at dette også har resultert i at en del zip-filer har blitt
  slettet, som muligens inneholdt måledata. Backup er tilgjengelig fra Boye:
sed -e'/Statuskoder/d' ../not_gs2.txt | while read path; do rm -v "$path"; done

* Noen av mappene inneholder duplikater av filer. Disse ble identifisert vha:
$ find . -type f | while read path; do base="`basename \"$path\"`"; find . -name "$base" | while read again; do if [ "$again" != "$path" ]; then echo -e "Duplicate of \n\t$path in\n\t$again";  fi; done; done

* En del av filene viste seg å inneholde binære data. Disse ble identifisert
  vha programmet find-binary (se avsnittet "preprosessering og konvertering..."
  for info om filene med temperaturdata):
$ ../find-binary -v <gs2.txt


* Forskjellige filer inneholder forskjellig nøkkel for å betegne EIA-nummeret. En
oversikt over alle nøkler ble derfor generert (etter sletting av filer med binært
innhold):
$ cat ../gs2.txt | while read f; do grep '^#' "$f"; done | sed -e's/=.*//' >../tags.txt 

* En oversikt over unike nøkler ble generert (sort | uniq fungerte ikke):
$ python unique.py <tags.txt >tags_unique.txt

* Basert på lista over unike nøkler, ble det funnet to forskjellige nøkler som
  inneholdt EIA-numre: #Reference og #Installation. Disse ble deretter
  anonymisert (dette må skje i en enkelt operasjon, for at hvert EIA-nummer
  skal få en unik erstatning på tvers av alle GS2-filene):
$ ../anonymize-gs2 <../gs2.txt

* Mappa "Malvik" inneholdt relativt lite data, i et litt annet format enn de
  andre filene, og med navngitte boligeiere. Mappa "Istad" inneholdt litt
  ymse. Disse mappene ble i sin helhet slettet, med antakelse om at vi har nok
  data å ta av.
==================================================



OM GS2-FORMATET OG INNLESING
--------------------------------------------------

GS2-formatet er tekstfiler bestående av nøkkel-verdi par. Hver nøkkel begynner
med en enkelt skigard (#). Nøkkel og verdi skilles med et likhetstegn (=). I
tillegg brukes doble skigarder med en etterfølgende tittel til å skille mellom
seksjoner (f.eks. "##Start-message" og "##Time-series").  Formatet er rimelig
selvforklarende, så her har jeg bare notert det som ser ut til å være
ustandard, eller som kan gjøre innlesingen trøblete.

Det ser ikke ut til å være helt konsekvent bruk av linjeskift (CR/LF).

I de fleste tilfellene ligger nøkkel og verdi på samme linje, men ikke
alltid. Spesielt er dette ofte ikke tilfelle med "Value"-nøkkelen (mao noen
ganger ligger alle målingene på samme linje, andre ganger er det brukt en linje
per måling). Her brukes imidlertid "større enn" og "mindre enn" (à la xml) for
å avgrense verdien: #Value=<...>.

Jfr anonymize_gs2.cpp, så brukes flere forskjellige nøkler til å identifisere
en måler/bruker/husstand. Noen bruker "Installation", andre bruker "Reference",
og atter andre bruker "Plant". Utfordringen er at i en enkelt fil kan flere av
disse nøklene finnes. For eksempel kan man finne følgende:

#Installation=1017766944        
#Plant=3

Imidlertid ser det ut til at det bare var Malvik som brukte "Plant", og disse
filene er nå slettet (jfr grep "^#Plant" <gs2files> | sort | uniq).

Mange (alle?) av filene inneholder en nøkkel "GMT-reference" i starten, noe som
kanskje kan brukes til å korrigere for potensielt manglende sommertid i
målingene?

De faktiske målingene ligger i nøkkelen kalt "Value". Hver måling består av
opptil tre elementer, adskilt med skråstrek: måleverdien, tidsstempel og
statuskode. Her er noen eksempler:

fra Istad: 0//X 
fra Buskerud:  1.285//1
fra Hafslund:  73.12//E
eller: 1//
Malvik brukte 2.5/2007-04-01.01:00:00/1. Uvisst om også andre gjør dette.

Under "Buskerud" ligger en fil kalt "Statuskoder.txt". Der står følgende:
Statuskoder:
1=Automatisk stipulert
0=Ok
99=Manuelt stipulert

Det er uvisst hva 1 og 0 betyr hos andre nettselskaper, hva X og E betyr, og om
det finnes enda flere mystiske statuskoder.



PREPROSESSERING OG KONVERTERING TIL ANDRE LAGRINGSFORMAT
--------------------------------------------------

I skrivende stund bruker vi primært de dataene som har temperaturmålinger, hvilket
betyr en stor andel av filene i mappa "Skagerak". Disse ble funnet vha:
$ ./gs2-grep.sh -l Celsius >gs2_with_temperatures.txt

I følge Nicolai Feilberg v/SINTEF Energi, er de første tallene i filnavnene
antakelig en indikasjon på hvordan målerne er lokalisert geografisk. For
eksempel kan disse betegne avgangen GS2-filen representerer. I noen tilfeller
ser det ut til at temperaturmåleren har feilet, men lastmålingene er fortsatt
til stede. Dette ble avdekket ved å gjøre følgende:
   1. Finn alle filene i Skagerak-mappa:
       $ find "`pwd`/../../../../data/sintef/raw/Skagerak" -iname '*.exp' -or -iname '*.gs2' >gs2_skagerak.txt
   2. Skill dem som allerede har temperaturer fra dem som ikke har det:
       $ cat gs2_skagerak.txt | while read f; do if [ -z "`grep "$f" gs2_with_temperatures.txt`" ]; then echo "X $f"; else echo "+ $f"; fi; done >files_with_without_temp_from_skagerak.txt

Med andre ord printet jeg ut en liste over alle filene i Skagerak-mappa, hvor
de med temperatur var indikert. Her kom det fram at temperaturmålingene finnes
i filene som begynner på 202_. Det endelige utvalget av filer ble derfor:
$ sed -n -e'/\/202_/p' gs2_skagerak.txt >gs2_for_prediction.txt

En diff av gs2_for_prediction.txt og gs2_with_temperatures.txt viste at alle
temperaturfilene ble med, samt ytterligere ca 25 filer.

Blant de utvalgte filene var det noen få som inneholdt binære data. I disse
filene var begynnelsen ok, mens slutten var korrumpert. De binære dataene ble
slettet, men resten ble beholdt. Filene det gjaldt var:
202_4_11_2006051423_2006052123_20060627081333000_1076011.exp
202_4_11_2006052823_2006060423_20060627081853000_1076032.exp
202_4_11_2006061123_2006061823_20060627082327000_1076046.exp

En "wc 202_4_11_20060*" viser at mye av filene er mistet, men i det minste er
temperaturmålingene beholdt.

HDF5 er betydelig raskere enn sqlite, men sqlite kan muligens ha noen fordeler
i forhold til brukervennlighet hvis man bare skal ha ut
enkeltverdier. sqlite-koden ble skrevet først, men vi gikk over til HDF5 fordi
ytelsen på sqlite var alt for dårlig.

AUTOMATISERT PREPROSESSERING OG UTVELGELSE:
--------------------------------------------------

Etter å ha gjort skrittene over, kjøres først skriptet "preprocess_gs2.py" for
å konvertere dataene til per-bruker tidsserier som lagres i HDF5, og deretter
"select_meters.py" for å velge ut målere til bruk i
vask&prediksjons-eksperimentene:
$ python preprocess_gs2.py
$ python select_meters.py


