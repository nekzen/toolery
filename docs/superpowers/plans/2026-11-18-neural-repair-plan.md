# Plan — réparer Neural-v1-24B

Rédigé le 2026-11-18. Document de travail : on le met à jour à chaque étape.

## But

Corriger les travers que la fusion a imposés à Neural **en gardant sa voix**.
Si la voix ne justifiait pas ce travail, on utiliserait le modèle de base ou un
autre (Gemma 4, Pale-Sonneteer) : c'est la seule raison d'entraîner plutôt que
de choisir.

Hors du premier entraînement : étirer la fenêtre de contexte efficace. On y
reviendra seulement si l'étape 2 montre un mur net et que l'étape 4 a marché.

## Ce que les mesures disent déjà (rp_long_context, 16k, 30 essais)

| Défaut | Neural | Sa base (MS3.2) | Avec le rappel | Lecture |
|---|---|---|---|---|
| Parler à la place de la joueuse (harbor-road, leak-ledger) | 20 | 9 | 3 | habitude, propre à la fusion |
| Parler à la place de la joueuse (snowline, player-voice) | 24 | — | 1 | habitude, confirmée sur une 2e histoire |
| Un inconnu nomme la protagoniste (harbor-road) | 22 | 19 | 6 | habitude, commune à la famille |
| Une information confiée fuit (snowline, carte normale ou avec le secret) | 6 / 1 | — | 3 | rare ; la carte n'y est pour rien |
| Répétitions (4 longueurs, 10 essais) | ~1 % de 4-grammes | — | — | aucun problème |
| Rétention d'un fait du début, jusqu'à 43k | 2 oublis sur 120 | — | — | tient |

Anomalie à surveiller : avec une note qui nomme un fait (`-told`), Neural en
fait une intrigue au lieu de la suivre (un messager envoyé par Hesper). Hypothèse :
il traite une consigne hors récit comme de la matière à raconter. Lié peut-être
aux consignes OOC absorbées signalées dans le brief.

Aucune de ces mesures ne ressemble à une limite du modèle.

### Hors RP : les outils (run complet à 65k, 5 essais, `--timeout-scale 8 --budget-slack 2`)

Avec un gabarit qui montre les outils dans le format natif de Mistral 3.2
(`[AVAILABLE_TOOLS]`), Neural obtient 28 % (sa base : 67,5 %). Sélection
d'outil 8 % contre 98 %, agentique 2,5 %, enchaînements 0 %. Il égale ou
dépasse sa base hors outils : suivi d'instructions 93 %, écriture créative 91 %.

Deux habitudes expliquent l'écart :

1. **Le format** : il écrit ses appels en JSON dans un bloc markdown, arguments
   encodés en chaîne, et n'émet jamais `[TOOL_CALLS]`. llama-server n'y voit
   que du texte. Avec `--text-tool-calls` (format pardonné, mesure seulement) :
   45 % au lieu de 28 %.
2. **Annoncer au lieu d'agir** : les 376 `wrong_tool` qui restent sont tous
   « I'll call the `get_weather` function with… », suivis d'un arrêt, sans
   appel. L'outil et les arguments annoncés sont les bons : le jugement tient,
   c'est le passage à l'acte qui manque.

Le gabarit livré dans le GGUF de Neural était celui des premières versions de
Mistral Small 3, sans aucune gestion des outils. Le gabarit corrigé
(`neural.jinja`, hors de ce dépôt) est un prérequis de tout ce qui suit.

## Étapes

### 1. Vérifications gratuites

- **Gabarit de chat de Neural sous llama-server.** Neural est nettement plus
  solide en agentique sous KoboldCpp. Comparer le prompt rendu par
  `/apply-template` avec celui de KoboldCpp. Un gabarit fautif se corrige sans
  toucher aux poids (voir Magistral 2509 : le gabarit seul a faussé un run
  entier).
- **Toolery complet à 65k** sur Neural et sur MS3.2-2506 : outils, code,
  instructions, efficacité. Il dit ce que la fusion a coûté ailleurs, et c'est
  le garde-fou contre les régressions de l'étape 4.

Livrable : un tableau Neural / MS3.2 par catégorie, et le gabarit retenu.

### 2. Mesurer le mur des 40k

- Remplir harbor-road jusqu'à ~56k tokens (aujourd'hui 43k).
- Lire la rétention, l'OOC et la répétition au-delà de 40k, sur Neural et MS3.2.
- Si le mur est net : prévoir côté inférence un résumé glissant et une mémoire
  (« roll summary + auto-RAG »), qui garde la fenêtre réelle sous 40k.

Livrable : le tableau à 8k / 16k / 32k / ~56k, et une décision sur la
longévité du contexte.

### 3. Le juge

- Une passe séparée sur les traces enregistrées, comme `rescore`. Jamais pendant
  un run.
- Sur une autre machine (dixie ou flatline), avec un modèle d'une autre famille
  que Mistral (par exemple Gemma 4 31B), à température 0.
- Sortie JSON avec une citation mot pour mot de la réponse. Un verdict dont la
  citation n'est pas dans la réponse est rejeté.
- Étalonnage d'abord : les réponses étiquetées à la main dans
  `tests/scenarios/test_rp_long_context.py`. On mesure l'accord avant de lui
  faire confiance.
- Usages : trancher les cas ambigus (la lettre inventée, les fuites
  « justifiées ») et surtout trier les paires de l'étape 4.

Livrable : `toolery judge`, et son taux d'accord avec les étiquettes.

### 4. Les travers : paires DPO et LoRA

**Les données ne viennent jamais du banc d'essai.** harbor-road et snowline
restent des tests.

- Des débuts d'histoires neufs et variés : genres, registres, tours de joueur
  à la première et à la deuxième personne, longueurs de 2k à 16k.
- Pour chaque scène : Neural génère **avec** la note de rappel (le choisi) et
  **sans** (le rejeté), même graine. Le juge garde les paires où le choisi est
  propre et le rejeté fautif.
- Les deux réponses viennent de Neural : il apprend à faire sans note ce qu'il
  fait déjà avec, dans sa propre voix.
- Cibles :
  1. s'arrêter là où le joueur doit parler ou agir ;
  2. ne pas faire nommer le protagoniste par qui ne le connaît pas ;
  3. traiter une note d'auteur ou une consigne OOC comme une instruction.
  4. appeler un outil dans le format natif, `[TOOL_CALLS]…[ARGS]…`, jamais en
     JSON dans le texte ;
  5. appeler l'outil au lieu d'annoncer qu'on va le faire.
- Pour les cibles 4 et 5, les exemples choisis viennent de **MS3.2-2506** (98 %
  en sélection d'outil), sur des tâches avec outils écrites pour l'occasion,
  jamais celles de toolery ; les rejetés sont les réponses de Neural aux mêmes
  tâches. Un seul format, le même gabarit (`neural.jinja`) à l'entraînement
  et à l'inférence.
- Volume de départ : 500 à 2 000 paires, dont une partie en contexte long.
  À ajuster d'après le premier entraînement, pas avant.
- Entraînement : LoRA avec Unsloth (AdamW 8 bits), DPO. Vérifier d'abord que
  la version d'Unsloth prend en charge le GB10.
- Évaluation : toolery complet avant et après, RP et le reste.

Critères de réussite, outils : sélection d'outil et agentique proches de
MS3.2-2506 en run `raw` normal (sans `--text-tool-calls`), mêmes réglages.

Critère de réussite, RP : Neural entraîné, **sans note**, atteint les taux de
Neural **avec** la note (parler à la place de la joueuse ≤ 3/30 sur les deux
histoires), sans perte notable dans les autres catégories, ni dans la voix
(lecture à l'œil d'un échantillon de réponses).

### 5. La longévité du contexte (plus tard, peut-être)

Seulement si l'étape 2 montre un mur et que l'étape 4 a marché. Demande des
données longues dont la réponse dépend du début, et des entraînements à 32-64k
de longueur de séquence : cher et incertain sur un GB10.

## Questions ouvertes

- Quel modèle pour le juge, sur quelle machine ?
- D'où viennent les débuts d'histoires de l'étape 4 : écrits, générés par un
  autre modèle, ou tirés de tes propres sessions ?
- Comment juger la voix autrement qu'à l'œil ? (Peut-être un petit test
  à l'aveugle : réponses de Neural avant/après, mélangées.)

## Journal du 2026-11-19 : les outils sous llama-server

- **Gabarit** : celui du GGUF de Neural (hérité de la base « text-only ») ignore
  les outils. `neural.jinja` (hors de ce dépôt) les rend dans le format natif.
- **Identifiants d'appel** : llama.cpp en fabrique de 32 caractères, Mistral a
  été entraîné sur 9. Le gabarit officiel lève une exception et llama.cpp ne
  peut plus construire son analyseur d'appels ; `neural.jinja` v2 garde les 9
  derniers caractères.
- **Tokenizer** : la conversion « text-only » a perdu le nom des jetons 32
  (`[ARGS]`) et 33 (`[CALL_ID]`), appelés `<SPECIAL_32>` et `<SPECIAL_33>`. Le
  GGUF d'Unsloth (version d'origine) les a. `neural.jinja` v3 les écrit
  `<SPECIAL_32>`/`<SPECIAL_33>` ; le tokenizer bf16 de Neural est corrigé
  (`neural-forge/forge/tokenizer_fix.py`, poids intacts).
- **Appels multiples** : la grammaire de llama.cpp pour les appels parallèles
  pousse MS3.2 dans un second appel et le rend coupé à `{`. Corrigé par
  `--no-parallel-tool-calls` (toolery) et `parallel_tool_calls: false` (forge).
- **Neural, servi correctement** : 32,1 %, **zéro appel natif sur 1 000
  essais** ; 290 annonces sans appel, 171 appels en JSON dans le texte, 79
  réponses sans outil (souvent inventées).
- **MS3.2 text-only** boucle encore en température 0 (8 192 tokens, plafond
  atteint) au milieu d'appels d'outils.
- Mistral ne publie pas de gabarit Jinja : le format officiel est celui de
  mistral-common. Tous les gabarits (Unsloth, llama.cpp, `neural.jinja`) sont
  des reconstructions.

### Décision ouverte : les outils dans la v1 ?

1. **v1 sans les outils** (recommandé) : entraîner d'abord le RP (cibles A, C,
   D), où les preuves et la méthode sont solides. Les outils plus tard.
2. **Les outils plus tard, par une autre voie** : la version d'origine de
   MS3.2 (GGUF d'Unsloth) obtenait 67,5 % en septembre. Une nouvelle fusion
   partant de la version d'origine, plutôt que de la « text-only », pourrait
   rendre les outils à moindre coût qu'une réparation par LoRA.
3. **Les outils dans la v1** : seulement après avoir vérifié le format exact
   avec mistral-common, puisque les données doivent être écrites dans ce
   format.
