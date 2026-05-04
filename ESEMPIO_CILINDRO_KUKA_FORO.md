# Guida utente - Datum C come foro decentrato

## Cosa misura il tastatore

L'operatore deve tastare:

1. punti sul cilindro esterno;
2. punti sul piano superiore;
3. punti sul foro decentrato, oppure il centro foro gia calcolato.

Esempio tastatura:

```text
Cilindro esterno: 8 punti sulla circonferenza
Piano superiore: 5 punti ben distribuiti
Foro decentrato: 8 punti sul foro
```

Il software usa:

```text
centro cilindro esterno = origine reale
piano superiore = Z reale
direzione centro cilindro -> foro = X reale
```

## Come preparare il CAD

Esempio pezzo alto 50 mm, foro 30 mm verso `+X`.

Nel CAD conviene mettere:

```text
Centro cilindro = 0, 0, 0
Piano superiore = Z 50
Centro foro = 30, 0, 0
```

Nel file/import:

```text
CENTRO_CILINDRO = 0 0 0
CENTRO_FORO = 30 0 0
Z_PIANO = 50
```

`CENTRO_FORO = 30 0 0` significa che il foro definisce la direzione `+X` del pezzo.

## Esempio pratico

Nel CAD:

```text
cilindro centrato in 0,0
piano superiore a Z=50
foro decentrato a X=30, Y=0
```

Nel software:

```text
Tipo Datum C = Foro decentrato / cilindro piccolo
Centro cilindro CAD = 0, 0, 0
Centro datum C CAD = 30, 0, 0
Quota piano superiore CAD = 50
```

Qui non serve `Inverti direzione Datum C reale`: la direzione e definita dal vettore centro cilindro -> centro foro.

## File KUKA esempio

Usare:

```text
ESEMPIO_CILINDRO_KUKA_FORO.txt
```
