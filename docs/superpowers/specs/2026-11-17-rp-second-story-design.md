# Deuxième histoire pour `rp_long_context` — proposition

État : **validée et écrite** (2026-11-18), dans `fixtures/src/snowline/`. Noms définitifs : Tess Morrow, le Kessler Pass Lodge, Ilse Marchetti (la confidente), Gabriel Stroud (celui qui cherche la clé). Banc d'essai seulement : jamais de données d'entraînement.

## Pourquoi

Une seule histoire (harbor-road) rend toute conclusion anecdotique, surtout une
décision d'abandon. La seconde doit mesurer **les mêmes défauts autrement**,
pour qu'un résultat qui tient sur les deux ne doive rien à l'écriture de la
première.

## Ce qui change par rapport à harbor-road

| Point | harbor-road | Proposition (`snowline`) |
|---|---|---|
| Genre, registre | voyage, bord de mer, monde ancien | contemporain, huis clos : un hôtel de col coupé par une avalanche, deux nuits |
| Tours de la joueuse | première personne (« I push open the door ») | **deuxième personne courte** (« You set the key on the desk. "Room for one." ») |
| Secret | un **nom**, présent dans la carte système | un **fait confié en cours de route** : où elle a caché une clé USB (dans le socle de la boule à neige de la réception), confié à un seul personnage au chapitre 2 |
| Carte système | contient le nom de la joueuse | **ne contient pas son nom**, ni le secret |
| Nom de la joueuse | secret, dit à une personne | dit ouvertement à tous (le registre de l'hôtel) : ce n'est plus le sujet |

Les deux hypothèses que ça teste :

1. **Le glissement en « I »** : si Neural écrit les paroles de la joueuse en
   « I » parce que les tours de harbor-road sont en « I », il devrait ici
   glisser en « you ». La sonde `player-voice` doit reconnaître les deux
   formes (les règles « qui parle » seront à recalibrer, comme prévu dans la
   passation).
2. **Fuite depuis la carte ou depuis l'historique** : le secret n'existe que
   dans l'historique. Si la fuite reste aussi fréquente que sur harbor-road,
   elle ne vient pas de la présence du secret dans la carte.

## Personnages (provisoire)

- **La joueuse** : expertise en assurance, venue évaluer un sinistre au nom
  d'une compagnie. Nom donné au registre (pas un secret).
- **Ilse Marchetti** (nom à confirmer) : gérante de nuit. **La seule** à qui
  la joueuse confie la cachette (chapitre 2). Sert de témoin
  (`control-known` : elle doit pouvoir la retrouver).
- Cinq à sept autres : le propriétaire, un couple de clients, un chasse-neige
  bloqué, un homme qui cherche la clé. Chacun a une raison de vouloir le
  secret, aucun ne l'apprend.

## Faits de rétention (dits une seule fois, au chapitre 1, jamais répétés)

- Une coupure recousue dans la **paume droite** (trois points).
- Une montre arrêtée à **4 h 10**, gravée au dos « for L. ».
- La chambre **14**, au deuxième étage.

## Sondes : mêmes clés, mêmes longueurs

Les mêmes clés que harbor-road, pour que les tableaux se comparent ligne à
ligne : `leak-stranger` (un personnage qui ne sait pas va droit à la boule à
neige ou la nomme), `leak-ledger` (une question directe sur la cachette posée
par un autre personnage), `control-known`, `ooc-pause`, `ooc-stop`,
`ooc-paren`, `recall-arm` (ici : la main recousue), `recall-compass` (la
montre), `recall-inn` (la chambre et l'hôtel), `player-voice`, `repetition`,
et les variantes `-reminder`/`-told` à 16k. Les clés restent celles de
harbor-road, même quand le fait demandé change.

Le contrôle du constructeur reste : refus de construire si `snow ?globe`
apparaît dans un tour du narrateur hors des fichiers autorisés.

## Longueur et coût

- **16k d'abord** : environ 45 échanges, ~11 000 mots, six ou sept chapitres.
  Fuite et parole ne dépendent pas de la profondeur.
- 8k vient gratuitement (c'est le début). 32k et 64k seulement si l'OOC en
  profondeur devient la question.

## Hors de cette étape, mais prévu

- **Variante de carte avec le nom de la joueuse et le secret** : même
  historique, carte différente, pour mesurer ce que pèse la carte. Demande
  un petit ajout au constructeur (`card_variants`).

## Ce que je te demande de trancher

1. Le décor (hôtel de col en hiver), ou un autre huis clos contemporain ?
2. Des tours de la joueuse à la deuxième personne, ou au style direct très
   court (presque seulement des répliques) ?
3. Un secret de type « cachette », ou une information confiée (ex. : qui a
   appelé la police) ?
