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
