# Guida utente - Datum C come piano laterale

## Cosa misura il tastatore

L'operatore deve tastare:

1. punti sul cilindro esterno;
2. punti sul piano superiore;
3. punti sulla faccia laterale scelta come Datum C.

Esempio tastatura:

```text
Cilindro esterno: 8 punti sulla circonferenza
Piano superiore: 5 punti ben distribuiti
Piano laterale: 4 o 5 punti sulla stessa faccia laterale
```

Il software usa:

```text
centro cilindro esterno = origine reale
piano superiore = Z reale
normale del piano laterale = direzione X reale
```

## Come preparare il CAD

Esempio pezzo alto 50 mm.

Nel CAD conviene mettere:

```text
Centro cilindro = 0, 0, 0
Piano superiore = Z 50
```

Se la faccia Datum C e il piano `X = 20`, quella faccia non passa dal centro cilindro. Va bene: il software usa solo la direzione della faccia.

Nel file/import:

```text
CENTRO_CILINDRO = 0 0 0
Z_PIANO = 50
DIREZIONE_DATUM_C = +X
```

`+X` significa: la normale CAD della faccia laterale punta verso `+X`.

## Esempio pratico

Nel CAD:

```text
cilindro centrato in 0,0
piano superiore a Z=50
faccia laterale Datum C a X=20
```

Nel software:

```text
Tipo Datum C = Piano laterale
Direzione CAD Datum C = +X CAD
```

La quota `X=20` non si inserisce. Serve solo per identificare la faccia nel CAD.

## Quando usare Inverti direzione Datum C reale

Una faccia ha sempre due normali possibili:

```text
normale verso +X
normale verso -X
```

Se il risultato viene girato di 180 gradi attorno alla Z, attivare:

```text
Inverti direzione Datum C reale
```

Non cambia il CAD. Cambia solo il verso della normale reale calcolata dalla tastatura.

## File KUKA esempio

Usare:

```text
ESEMPIO_CILINDRO_KUKA_PIANO_LATERALE.txt
```
