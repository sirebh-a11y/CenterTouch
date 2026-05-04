# Guida rapida - cilindro con Datum C

## Regola comune

Il software costruisce sempre il riferimento cosi:

```text
Origine reale = centro del cilindro esterno sul piano superiore
Z reale       = normale del piano superiore tastato
X reale       = direzione scelta dal Datum C
Y reale       = calcolata automaticamente
```

L'operatore misura il pezzo reale. Il CAD deve essere impostato con la stessa logica.

## Come mettere il pezzo nel CAD

Modo consigliato:

```text
Centro cilindro CAD = 0, 0, 0
Piano superiore CAD = Z = altezza pezzo
```

Esempio con pezzo alto 50 mm:

```text
CENTRO_CILINDRO = 0 0 0
Z_PIANO = 50
```

Il software usa come origine nominale il centro cilindro alla quota del piano superiore:

```text
Origine nominale = 0, 0, 50
```

## Caso 1 - Datum C come foro decentrato

Tastatura:

- punti sul cilindro esterno;
- punti sul piano superiore;
- punti sul foro decentrato, oppure centro foro.

CAD:

```text
CENTRO_CILINDRO = 0 0 0
CENTRO_FORO = 30 0 0
Z_PIANO = 50
```

Significa: il foro e 30 mm verso `+X` rispetto al centro cilindro. La X del pezzo va dal centro cilindro al foro.

## Caso 2 - Datum C come piano laterale

Tastatura:

- punti sul cilindro esterno;
- punti sul piano superiore;
- punti sulla faccia laterale scelta come Datum C.

CAD:

Esempio: la faccia scelta e il piano `X = 20`. Quella faccia non passa dal centro, ma la sua normale punta verso `+X`.

Nel software:

```text
Tipo Datum C = Piano laterale
CENTRO_CILINDRO = 0 0 0
Z_PIANO = 50
DIREZIONE_DATUM_C = +X
```

La quota `X = 20` non si inserisce: per questo caso il software usa solo la direzione della faccia, non la sua distanza dal centro.

Attenzione: se una faccia "corre lungo X", non vuol dire che la sua normale sia `+X`. Bisogna scegliere la direzione perpendicolare alla faccia.

## Caso 3 - Datum C come linea / asse

Tastatura:

- punti sul cilindro esterno;
- punti sul piano superiore;
- almeno 2 punti sulla linea, cava, spigolo o asse scelto.

CAD:

Esempio: una cava e parallela a `+Y`, anche se sta lontana dal centro, per esempio a `X = 20`.

Nel software:

```text
Tipo Datum C = Linea / asse
CENTRO_CILINDRO = 0 0 0
Z_PIANO = 50
DIREZIONE_DATUM_C = +Y
```

La linea non deve passare dal centro cilindro. Conta solo la sua direzione.

## Cosa significa +X, -X, +Y, -Y

Nel software devi dire quale direzione CAD rappresenta il Datum C.

Esempi:

```text
Foro a destra del centro CAD        -> +X
Foro a sinistra del centro CAD      -> -X
Linea/cava che sale lungo Y CAD     -> +Y
Linea/cava che scende lungo Y CAD   -> -Y
Faccia X = 20 con normale verso +X  -> +X
Faccia X = -20 con normale verso -X -> -X
```

## Quando usare "Inverti direzione Datum C reale"

Serve solo per `Piano laterale` e `Linea / asse`.

Motivo: una linea e un piano possono avere due versi validi.

Esempio linea:

```text
Punto 1 -> Punto 2 = +Y
Punto 2 -> Punto 1 = -Y
```

Esempio piano laterale:

```text
La stessa faccia puo avere normale +X oppure -X
```

Se dopo il calcolo il pezzo risulta girato di 180 gradi attorno alla Z, attiva:

```text
Inverti direzione Datum C reale
```

Non cambia il CAD. Cambia solo il verso della direzione reale calcolata dalla tastatura.

## File esempio e spiegazione

```text
ESEMPIO_CILINDRO_KUKA_FORO.txt            -> punti Datum C = foro decentrato
ESEMPIO_CILINDRO_KUKA_FORO.md             -> spiegazione foro decentrato

ESEMPIO_CILINDRO_KUKA_PIANO_LATERALE.txt  -> punti Datum C = piano laterale
ESEMPIO_CILINDRO_KUKA_PIANO_LATERALE.md   -> spiegazione piano laterale

ESEMPIO_CILINDRO_KUKA_LINEA.txt           -> punti Datum C = linea / asse
ESEMPIO_CILINDRO_KUKA_LINEA.md            -> spiegazione linea / asse
```
