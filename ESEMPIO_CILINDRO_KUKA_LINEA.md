# Guida utente - Datum C come linea / asse

## Cosa misura il tastatore

L'operatore deve tastare:

1. punti sul cilindro esterno;
2. punti sul piano superiore;
3. almeno 2 punti sulla linea scelta come Datum C.

La linea puo essere:

```text
spigolo
cava
guida
asse lineare
```

Il software usa:

```text
centro cilindro esterno = origine reale
piano superiore = Z reale
direzione della linea = X reale
```

La linea non deve passare dal centro cilindro.

## Come preparare il CAD

Esempio pezzo alto 50 mm.

Nel CAD conviene mettere:

```text
Centro cilindro = 0, 0, 0
Piano superiore = Z 50
```

Se la linea Datum C e una cava parallela a `+Y`, anche se sta a `X=20`, va bene.

Nel file/import:

```text
CENTRO_CILINDRO = 0 0 0
Z_PIANO = 50
DIREZIONE_DATUM_C = +Y
```

`+Y` significa: nel CAD la linea scelta va nel verso `+Y`.

## Esempio pratico

Nel CAD:

```text
cilindro centrato in 0,0
piano superiore a Z=50
cava lineare a X=20
cava orientata lungo +Y
```

Nel software:

```text
Tipo Datum C = Linea / asse
Direzione CAD Datum C = +Y CAD
```

La quota `X=20` non si inserisce. Serve solo per trovare la cava nel CAD.

## Quando usare Inverti direzione Datum C reale

Una linea ha due versi:

```text
punto 1 -> punto 2
punto 2 -> punto 1
```

Se il tastatore prende i punti nel verso opposto, il software puo vedere la linea come `-Y` invece che `+Y`.

Se il risultato viene girato di 180 gradi attorno alla Z, attivare:

```text
Inverti direzione Datum C reale
```

Non cambia il CAD. Cambia solo il verso della linea reale calcolata dalla tastatura.

## File KUKA esempio

Usare:

```text
ESEMPIO_CILINDRO_KUKA_LINEA.txt
```
