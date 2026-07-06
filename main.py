import os
import langchain_google_vertexai

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()



def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk (wym. /ˈi:lɒn ˈmʌsk/; ur. 28 czerwca 1971 w Pretorii[4]) – południowoafrykański przedsiębiorca, założyciel, współzałożyciel lub finansista przedsiębiorstw SpaceX, Tesla, Neuralink, X.com (część firmy PayPal), The Boring Company[5] oraz xAI. Pochodzi z Republiki Południowej Afryki, mieszka i pracuje w Stanach Zjednoczonych (posiada obywatelstwo południowoafrykańskie, kanadyjskie i amerykańskie). Dyrektor generalny i techniczny w SpaceX, dyrektor generalny i główny architekt w Tesla Inc. W styczniu 2021 został uznany najbogatszym człowiekiem świata przez magazyn „Forbes” i agencję Bloomberg﻿[w innych językach][6][7][8]. Od 28 października 2022 właściciel serwisu X (kiedyś „Twitter”). W okresie od 20 stycznia do 28 maja 2025 szef Departamentu Wydajności Rządu﻿[w innych językach] (ang. Department of Government Efficiency, DOGE) w drugim gabinecie Donalda Trumpa. Na dzień 16 czerwca 2026 roku, według magazynu „Forbes”, jego majątek szacowany jest na 1,3 biliona dolarów amerykańskich (USD)[1].

    Życiorys
    Dzieciństwo i edukacja
    Pochodzi z Południowej Afryki[9]. Urodził się i wychował w stołecznej Pretorii, w białej rodzinie. Jego ojciec, inżynier Errol Musk﻿[w innych językach], urodził się w Południowej Afryce jako syn Brytyjki i Południowoafrykańczyka, natomiast matka, modelka i dietetyczka Maye Musk, ma po ojcu amerykańskie pochodzenie i przyszła na świat w Kanadzie[10][11]. W 1950 wyjechała z rodzicami do Południowej Afryki[10]. Elon ma dwoje młodszego rodzeństwa: brata Kimbala﻿[w innych językach] i siostrę Toscę[11].

    Po rozwodzie rodziców w 1980 mieszkał głównie z ojcem[10]. Kiedy miał 10 lat, dostał pierwszy komputer i nauczył się programować[12] (według innego źródła nauczył się programować w wieku 13 lat)[13]. Dwa lata później sprzedał swój pierwszy program – grę komputerową Blastar za około 500 dolarów[12].

    Jako nastolatek uczęszczał do Pretoria Boys High School, którą ukończył w wieku 17 lat. Krótko później, częściowo z chęci uniknięcia obowiązkowej służby wojskowej w Południowoafrykańskich Siłach Obronnych (SADF), wyemigrował do Kanady, gdzie mieszkała rodzina jego matki[12]. Planował przeniesienie się do Stanów Zjednoczonych[14][15].

    W Kanadzie pracował u kuzyna na farmie w Swift Current, przy czyszczeniu kotłów w tartaku w Kolumbii Brytyjskiej i przy wyrębie lasów. Po dwóch latach przeniósł się do Toronto i pracował w dziale IT w banku, aplikując jednocześnie do Queen’s University. Opuścił Kanadę w 1992 roku po uzyskaniu stypendium na University of Pennsylvania. Tam uzyskał tytuł licencjata w dziedzinie ekonomii na wydziale Wharton Business School, po czym studiował tam jeszcze rok uzyskując tytuł licencjata w dziedzinie fizyki[16].

    Kariera
    W 1995 roku rozpoczął studia na wydziale fizyki w Stanford University, po czym po dwóch dniach je rzucił, aby założyć przedsiębiorstwo Zip2.

    Zip2
    W 1995 z bratem, Kimbalem, korzystając z pieniędzy zebranych od niewielkiej grupy inwestorów-aniołów[17] oraz z pomocą finansową ojca w wysokości 200 tys. dol[18], założył Zip2﻿[w innych językach], firmę zajmującą się oprogramowaniem sieciowym. Firma opracowała i sprzedała internetowy „przewodnik po mieście” dla branży wydawniczej[19]. Musk uzyskał kontrakty z „The New York Times” i „Chicago Tribune”[20] oraz przekonał zarząd do porzucenia planów połączenia się z CitySearch. Będąc w Zip2, Musk chciał zostać CEO; jednak nikt z członków zarządu na to nie pozwolił. Compaq przejął Zip2 za 307 milionów USD w gotówce i 34 miliony USD w opcjach na akcje w lutym 1999[21]. Musk otrzymał 22 miliony USD za swój siedmioprocentowy udział w sprzedaży[20].

    X.com i PayPal
    Główne artykuły: PayPal i PayPal Mafia.
    W marcu 1999 został współzałożycielem X.com, firmy świadczącej usługi finansowe i obsługującej płatności za pośrednictwem poczty elektronicznej, o wartości 10 milionów dol. Pieniądze na firmę uzyskał ze sprzedaży Zip2[20][22]. Rok później X.com połączyło się z firmą Confinity﻿[w innych językach][20], która była właścicielem PayPala, usługi transferu pieniędzy. Połączona firma skupiła się na usłudze PayPal i została przemianowana na „PayPal” w 2001. Wczesny wzrost PayPal był napędzany głównie przez wirusową kampanię marketingową, w której rekrutowano nowych klientów, gdy otrzymywali pieniądze za pośrednictwem usługi[23]. Musk został w październiku 2000 usunięty ze swojej funkcji dyrektora generalnego (choć pozostał w zarządzie) ze względu na nieporozumienia z pozostałym kierownictwem firmy. Musk chciał zmigrować serwery PayPala z systemu Unix do systemu Microsoft Windows, z czym nie zgadzał się zarząd[24]. W październiku 2002 serwis eBay przejął PayPal za 1,5 mld dol. w akcjach, z czego Musk otrzymał 165 mln dol. Przed sprzedażą Musk, który był największym udziałowcem spółki, posiadał 11,7% akcji PayPala[25].

    W lipcu 2017 kupił domenę x.com od PayPal za nieujawnioną kwotę, stwierdzając, że ma dla niego „wielką wartość sentymentalną”[26].

    SpaceX
    Główny artykuł: SpaceX.

    Musk oprowadzający prezydenta Baracka Obamę po stanowisku startowym rakiety Falcon 9.
    W czerwcu 2002 założył swoje trzecie przedsiębiorstwo, Space Exploration Technologies (SpaceX), którego został dyrektorem generalnym i technicznym. SpaceX zajmuje się projektowaniem i konstruowaniem[potrzebny przypis] rakiet nośnych dla pojazdów kosmicznych, kładąc nacisk na ich niski koszt i niezawodność. Celem, który postawił sobie Musk, jest 100-krotne zmniejszenie kosztów lotów w kosmos, między innymi poprzez wielokrotne starty tych samych egzemplarzy rakiet. Pierwszą opracowaną przez przedsiębiorstwo rakietą była Falcon 1, wystrzelona po raz pierwszy 24 marca 2006. Rakieta ta we wrześniu 2009 jako pierwsza prywatna rakieta w historii astronautyki umieściła satelitę na orbicie Ziemi. Drugą rakietą jest Falcon 9, wystrzelona po raz pierwszy 4 czerwca 2010. Jej głównym zadaniem jest wynoszenie na orbitę stworzonych również przez to przedsiębiorstwo statków kosmicznych Dragon, obsługujących zaopatrzenie Międzynarodowej Stacji Kosmicznej. SpaceX pracuje również nad konstrukcją statku kosmicznego Starship, który ma być w stanie zapewnić obecność stałej kolonii na Marsie[27], oraz rozbudowuje konstelację satelitów Starlink, dostarczającą Internet satelitarny.

    23 grudnia 2008 SpaceX zdobyła wart 1,6 miliarda dol. kontrakt na przeprowadzenie 12 lotów zaopatrzeniowych na Międzynarodową Stację Kosmiczną, przejmując rolę wahadłowców kosmicznych po ich wycofaniu z użytku w 2011. Początkowo pojazdy Dragon mają przewozić jedynie zaopatrzenie, pozostawiając transport astronautów statkom Sojuz. Augustine commission rekomendowała, aby w przyszłości transport astronautów przekazać komercyjnym przedsiębiorstwom, takim jak SpaceX[28].

    30 maja 2020 roku o godzinie 19:22 UTC (21:22 czasu polskiego) z platformy startowej na Florydzie wystartowała rakieta Falcon 9 z astronautami Robertem Behnkenem i Douglasem Hurleyem w ramach misji dostarczenia astronautów na Międzynarodową Stację Kosmiczną przez SpaceX[29]. 31 maja o godzinie 14:16 UTC (16:16 czasu polskiego), po kilkunastogodzinnym locie, kapsuła Dragon przycumowała do ISS, co zakończyło powodzeniem ten etap misji[30]. W jej ramach, astronauci pozostali na stacji przez kilka miesięcy, a następnie powrócili na Ziemię[29].

    Musk postrzega badania kosmosu jako istotny krok w rozprzestrzenieniu, a nawet przetrwaniu ludzkiej świadomości. Stwierdził, że kolonizacja innych planet może stanowić zabezpieczenie przed zagrożeniami dla przetrwania gatunku: „Może nas zniszczyć uderzenie planetoidy, superwulkan, jak również zagrożenia, których dinozaury nigdy nie doświadczyły: zaprojektowany wirus, nieumyślne stworzenie mikro-czarnej dziury, katastrofalne globalne ocieplenie albo jakaś jeszcze nieznana technologia. Ludzkość ewoluowała przez miliony lat, ale w ciągu ostatnich sześćdziesięciu lat broń atomowa dała nam możliwość unicestwienia siebie. Prędzej czy później musimy rozprzestrzenić się poza tę zielono-niebieską kulę, albo wyginąć”[31].

"""

    summary_template = """ 
    na podstawie informacji {information} o danej osobie:
    utwórz krótkie podsumowanie w, które zawiera:
    - imię i nazwisko osoby
    - datę i miejsce urodzenia
    - główne osiągnięcia zawodowe i przedsiębiorstwa, które założył oraz 2 ciekawostki na jego temat
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],template=summary_template)


    llm= ChatGoogleGenerativeAI(
        temperature=0, 
        model="gemini-3.5-flash", 
    )
    """

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    """
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
