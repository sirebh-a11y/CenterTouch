# Proposta modulo cilindro con foro decentrato

## Scopo

Il secondo modulo deve seguire la stessa logica del modulo gia esistente:

1. importare o inserire punti tastati;
2. fare fit geometrici sulle primitive;
3. costruire un frame reale;
4. costruire un frame nominale CAD con la stessa logica;
5. calcolare la trasformazione CAD -> reale;
6. produrre output `Translate` / `Rotate` per Meltio Space;
7. mostrare un report qualita con messaggi chiari per l'utente.

Il caso e quello del secondo schema in `datum.png`:

- Datum A: cilindro esterno;
- Datum B: piano superiore;
- Datum C: foro decentrato oppure piccolo cilindro decentrato.

## Decisioni metrologiche

### Direzione Z reale

La direzione `Z reale` deve arrivare dalla normale del piano superiore tastato.

Motivo: il cilindro esterno potrebbe essere tastato su due fasce troppo vicine. In quel caso usare il fit cilindro per stimare la direzione asse sarebbe metrologicamente instabile.

Quindi:

```text
Z reale = normale del piano superiore
```

Come nel modulo attuale, deve esistere una opzione per invertire il verso di `Z reale`.

### Cilindro esterno

Il cilindro esterno serve principalmente per trovare il centro del pezzo sul piano reale.

Non si usa il cilindro esterno per definire la direzione asse. L'asse del cilindro e assunto normale al piano superiore.

Procedura proposta:

1. si fitta il piano superiore;
2. si usa la normale del piano come `Z reale`;
3. si proiettano i punti tastati del cilindro esterno sul piano superiore o su un piano parallelo coerente;
4. si fa un fit cerchio nella base del piano;
5. il centro del cerchio diventa il centro cilindro reale;
6. l'origine reale e il centro cilindro proiettato/usato sul piano superiore.

In termini pratici:

```text
O reale = centro cilindro esterno sul piano superiore
```

### Piano superiore

Il piano superiore viene tastato come nel modulo attuale:

- minimo 3 punti;
- consigliati 4 o 5 punti;
- punti distribuiti bene sulla superficie.

Serve a:

- definire `Z reale`;
- fissare la quota dell'origine reale;
- proiettare i datum che non devono usare direttamente la quota tastata.

### Foro decentrato / piccolo cilindro

Il terzo datum serve a bloccare la rotazione attorno a `Z reale`.

Puo essere:

- foro interno decentrato;
- piccolo cilindro esterno decentrato;
- centro gia calcolato.

Regola di priorita:

```text
Se esistono punti tastati, usare i punti.
Se non esistono punti, usare il centro inserito/importato.
Se esistono entrambi, usare i punti e avvisare nel report che il centro importato e stato ignorato.
```

La direzione `X reale` proposta e:

```text
X reale = direzione da O reale al centro del foro/cilindro decentrato,
          proiettata sul piano normale a Z reale
```

Poi:

```text
Y reale = Z reale x X reale
```

Il sistema deve restare destrorso.

## Frame reale

La costruzione proposta e:

```text
O = centro cilindro esterno sul piano superiore
Z = normale piano superiore
X = direzione O -> centro foro/cilindro decentrato, proiettata sul piano superiore
Y = Z x X
```

Controlli da fare:

- `X` non deve essere quasi nullo;
- il foro/cilindro decentrato non deve essere troppo vicino al centro cilindro;
- il piano deve avere RMS accettabile;
- i punti cilindro devono produrre un cerchio stabile;
- il datum C deve produrre un centro stabile.

## Frame nominale CAD

Nella prima versione il CAD nominale richiede:

- centro cilindro nominale;
- quota piano superiore nominale;
- centro foro/cilindro decentrato nominale.

Si assume:

```text
asse cilindro nominale = normale al piano Z nominale
```

Quindi:

```text
O nominale = (centro cilindro nominale X/Y, Z_PIANO)
Z nominale = (0, 0, 1) oppure invertito da checkbox
X nominale = direzione O nominale -> centro foro/cilindro decentrato nominale
Y nominale = Z nominale x X nominale
```

## Trasformazione

La trasformazione resta la stessa del modulo attuale:

```text
R = real_frame.R @ nominal_frame.R.T
t = real_frame.origin - R @ nominal_frame.origin
```

L'output deve mantenere la stessa logica del modulo esistente:

- `Translate`;
- `Rotate`;
- scelta convenzione rotazioni da parte dell'utente;
- warning gimbal lock;
- matrice `R`;
- matrice omogenea `T`;
- dettagli fit;
- quality status.

## GUI proposta

La finestra deve essere simile al modulo attuale, senza cambiare stile.

### Dati reali da tastatura

Blocchi:

1. **Cilindro esterno reale**
   - modalita punti tastati;
   - in futuro eventuale centro gia calcolato;
   - tabella X/Y/Z;
   - bottoni aggiungi/rimuovi riga.

2. **Piano superiore reale**
   - tabella X/Y/Z;
   - stessa logica del piano attuale.

3. **Datum C reale**
   - tendina tipo datum:
     - `Foro decentrato / cilindro piccolo`;
     - `Piano laterale`;
     - `Linea / asse`;
     - `On demand / da definire`;
   - nella prima versione funziona solo `Foro decentrato / cilindro piccolo`;
   - modalita punti oppure centro gia calcolato;
   - se punti e centro esistono, usare punti.

### Dati nominali CAD

Campi:

- centro cilindro CAD;
- centro foro/cilindro decentrato CAD;
- quota piano superiore CAD;
- opzione inverti Z nominale.

### Opzioni

Come nel modulo attuale:

- inverti Z reale;
- inverti Z nominale;
- compensazione piano se serve;
- convenzione output rotazioni;
- soglie qualita.

## Testi guida per utente

### Cilindro esterno

```text
Metodo cilindro esterno:
- Inserire punti tastati sulla superficie cilindrica esterna.
- I punti servono a calcolare il centro del cilindro in sezione.
- La direzione asse non viene ricavata dal cilindro: viene presa dalla normale del piano superiore.
```

### Piano superiore

```text
Metodo piano superiore:
- Inserire 3, 4 o 5 punti ben distribuiti.
- La normale del piano definisce l'asse Z reale.
- Il piano fissa la quota dell'origine reale.
```

### Datum C

```text
Metodo foro/cilindro decentrato:
- Preferire punti tastati se disponibili.
- Se sono presenti sia punti sia centro, verranno usati i punti.
- Questo datum blocca la rotazione attorno all'asse Z.
```

### Varianti Datum C non ancora attive

```text
Questa variante datum richiede una procedura dedicata.
Per ora usare Foro decentrato / cilindro piccolo, oppure contattare Wire Trading e SiRe per definire la strategia di tastatura.
```

## Varianti Datum C da aggiungere

Le varianti devono mantenere la stessa struttura del modulo cilindro:

```text
O = centro cilindro esterno sul piano superiore
Z = normale piano superiore
X = definito dal Datum C scelto
Y = Z x X
```

Quindi cambia solo il modo in cui viene costruita la direzione `X`.

### Variante 1 - Piano laterale

Il Datum C e un piano laterale tastato.

#### Input reale

L'utente inserisce punti tastati sul piano laterale.

Il software:

1. fa il fit del piano laterale;
2. prende la normale del piano laterale;
3. proietta la normale sul piano superiore, quindi sul piano perpendicolare a `Z reale`;
4. normalizza la direzione;
5. usa quella direzione per costruire l'asse `X reale`.

Formula concettuale:

```text
n_laterale = normale fit piano laterale
X reale = proiezione di n_laterale sul piano normale a Z reale
```

Serve una opzione utente:

```text
Inverti direzione piano laterale
```

Motivo: la normale di un piano ha sempre due versi possibili. Il verso corretto dipende da come si vuole orientare il frame pezzo.

#### Input CAD

Per la prima versione non serve far inserire un piano CAD completo.

L'utente deve scegliere quale direzione nominale rappresenta quel piano laterale:

```text
Direzione nominale Datum C:
- +X CAD
- -X CAD
- +Y CAD
- -Y CAD
```

Il software usa questa direzione per costruire il frame nominale.

Limite attuale:

```text
La scelta +X / -X / +Y / -Y funziona solo se il piano laterale CAD e parallelo agli assi principali.
```

Se il piano laterale e inclinato rispetto agli assi CAD, manca ancora la gestione dedicata.

Soluzioni possibili da discutere prima di implementare:

1. **Angolo CAD nel piano XY**
   - L'utente inserisce l'angolo della normale del piano rispetto a `+X`.
   - Esempio: `45 gradi` significa normale a meta tra `+X` e `+Y`.
   - E piu semplice per l'utente se il CAD mostra chiaramente l'angolo.

2. **Vettore CAD manuale**
   - L'utente inserisce la normale CAD come `X Y Z`.
   - Esempio: `0.707 0.707 0`.
   - E piu generale, ma piu facile da sbagliare.

Raccomandazione provvisoria:

```text
Prima di aggiungere codice, scegliere una sola soluzione finale tra Angolo CAD e Vettore CAD manuale.
Per uso officina/robot, Angolo CAD nel piano XY sembra la scelta piu semplice se il datum e sempre sul piano superiore.
```

#### Testo guida utente

```text
Metodo piano laterale:
- Inserire punti tastati su una superficie laterale del pezzo.
- La normale del piano laterale definisce la direzione angolare attorno a Z.
- Se l'asse risulta girato di 180 gradi, usare Inverti direzione piano laterale.
- Nel CAD selezionare se quel piano corrisponde a +X, -X, +Y o -Y.
```

#### Controlli qualita

Il report deve includere:

- RMS fit piano laterale;
- max residual piano laterale;
- angolo tra normale piano laterale e `Z reale`;
- warning se il piano laterale e quasi parallelo al piano superiore;
- warning se la proiezione della normale laterale sul piano superiore e troppo piccola.

### Variante 2 - Linea / asse da punti

Il Datum C e una linea reale tastata o una direzione derivata da punti.

Non include, in questa fase, il fit asse foro/cilindro su piu sezioni. Quella sara una variante successiva.

#### Input reale

L'utente inserisce almeno 2 punti su una linea, spigolo, guida, scanalatura o riferimento direzionale.

Il software:

1. legge i punti linea;
2. se sono 2 punti, usa la direzione punto 1 -> punto 2;
3. se sono piu di 2 punti, fa un fit linea con PCA;
4. proietta la direzione sul piano superiore;
5. normalizza la direzione;
6. usa quella direzione come `X reale`.

Formula concettuale:

```text
d_linea = direzione fit linea
X reale = proiezione di d_linea sul piano normale a Z reale
```

Serve una opzione utente:

```text
Inverti direzione linea
```

Motivo: anche una linea ha due versi possibili.

#### Input CAD

Per la prima versione ci sono due modi possibili.

Modo semplice consigliato:

```text
Direzione nominale Datum C:
- +X CAD
- -X CAD
- +Y CAD
- -Y CAD
```

Modo avanzato futuro:

```text
Vettore linea CAD = X Y Z
```

Per la prima implementazione e meglio usare la tendina `+X / -X / +Y / -Y`, per evitare errori di inserimento vettore.

Limite attuale:

```text
La scelta +X / -X / +Y / -Y funziona solo se la linea CAD e parallela agli assi principali.
```

Se la linea e inclinata rispetto agli assi CAD, manca ancora la gestione dedicata.

Soluzioni possibili da discutere prima di implementare:

1. **Angolo CAD nel piano XY**
   - L'utente inserisce l'angolo della linea rispetto a `+X`.
   - Esempio: `30 gradi` significa linea inclinata di 30 gradi rispetto a `+X`.
   - E semplice se la linea e nel piano superiore o viene comunque usata solo come direzione proiettata.

2. **Vettore CAD manuale**
   - L'utente inserisce la direzione CAD come `X Y Z`.
   - Esempio: `0.866 0.500 0`.
   - E piu generale, ma piu delicato.

Raccomandazione provvisoria:

```text
Prima di aggiungere codice, discutere quale input finale adottare.
Per la prima versione completa, Angolo CAD nel piano XY e probabilmente piu chiaro; Vettore CAD manuale puo restare modalita avanzata.
```

#### Testo guida utente

```text
Metodo linea / asse:
- Inserire almeno 2 punti sul riferimento lineare.
- Con 2 punti viene usata la direzione punto 1 -> punto 2.
- Con piu punti viene calcolata la linea media.
- La direzione viene proiettata sul piano superiore.
- Se l'asse risulta girato di 180 gradi, usare Inverti direzione linea.
- Nel CAD selezionare se quella linea corrisponde a +X, -X, +Y o -Y.
```

#### Controlli qualita

Il report deve includere:

- numero punti linea;
- RMS o scostamento massimo dalla linea fit;
- warning se i punti sono troppo vicini tra loro;
- warning se la linea e quasi parallela a `Z reale`;
- warning se la proiezione della linea sul piano superiore e troppo piccola.

## Formato TXT proposto

Il formato deve restare vicino al file KUKA gia visto:

```text
# =========================
# CILINDRO ESTERNO - PUNTI TASTATI
# =========================
X Y Z
X Y Z

# =========================
# PIANO SUPERIORE - PUNTI
# =========================
X Y Z
X Y Z

# =========================
# FORO DECENTRATO - PUNTI TASTATI
# =========================
X Y Z
X Y Z

# =========================
# FORO DECENTRATO - CENTRO
# =========================
X Y Z

# =========================
# CAD NOMINALE CILINDRO
# =========================
CENTRO_CILINDRO = X Y Z
CENTRO_FORO = X Y Z
Z_PIANO = h
```

Note:

- i commenti con `#` sono ammessi;
- le righe `---` e firme finali devono essere ignorate;
- i numeri possono avere punto o virgola decimale;
- se sono presenti sia punti sia centro per il datum C, si usano i punti;
- il centro importato resta utile come confronto o dato informativo.

## Import KUKA / Renishaw

Per ora si definisce un formato TXT semplice da far produrre a KUKA.

Il parser futuro dovrebbe comunque essere robusto e accettare anche:

```text
123.456 78.900 12.345
X=123.456 Y=78.900 Z=12.345 A=0.0 B=0.0 C=0.0
{E6POS: X 123.456, Y 78.900, Z 12.345, A 0.0, B 0.0, C 0.0}
```

Per i fit geometrici si useranno solo `X`, `Y`, `Z`.

Orientamenti robot `A`, `B`, `C` e altri assi esterni non devono influenzare il calcolo del frame pezzo.

## Report qualita

Il report dovrebbe includere:

- RMS e max residual piano superiore;
- RMS e max residual fit cerchio cilindro esterno;
- raggio cilindro esterno misurato;
- RMS e max residual foro/cilindro decentrato;
- raggio datum C misurato;
- distanza reale centro cilindro -> centro foro/cilindro decentrato;
- confronto con distanza nominale CAD;
- warning se il datum C e troppo vicino all'origine;
- warning se i punti del cilindro esterno sono poco distribuiti;
- warning se il piano e instabile;
- warning se vengono importati sia punti sia centro e il centro viene ignorato.

## Rischi tecnici

1. **Punti cilindro poco distribuiti**
   - Se i punti sono concentrati su pochi angoli, il centro cilindro puo diventare instabile.

2. **Fasce cilindro troppo vicine**
   - Non devono essere usate per definire l'asse.
   - La direzione asse arriva dal piano superiore.

3. **Datum C vicino al centro**
   - Se il foro/cilindro decentrato e troppo vicino al centro cilindro, la direzione X diventa instabile.

4. **Compensazione tastatore**
   - Per ora si usa la stessa logica del modulo esistente.
   - Eventuali compensazioni piu complete andranno discusse dopo.

5. **Formato file**
   - Va definito e testato con un file generato da KUKA.
   - L'esempio TXT allegato serve come base di prova.

## Implementazione proposta

### Versione 1

- aggiungere il modulo cilindro dietro il pulsante `Trasforma cilindro`;
- mantenere aspetto e logica del modulo attuale;
- implementare:
  - punti cilindro esterno;
  - punti piano superiore;
  - datum C come foro/cilindro decentrato, con punti o centro;
  - CAD nominale con centro cilindro, centro foro, quota piano;
  - output rotazioni selezionabile come nel modulo attuale;
  - report qualita.

### Versione 2

- varianti Datum C:
  - piano laterale;
  - linea/asse;
  - procedure on demand;
- parser KUKA/Renishaw avanzato;
- eventuali compensazioni tastatore specifiche per cilindri/fori;
- confronto piu completo tra centri importati e centri calcolati.
