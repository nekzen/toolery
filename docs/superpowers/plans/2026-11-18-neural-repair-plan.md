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
- Volume de départ : 500 à 2 000 paires, dont une partie en contexte long.
  À ajuster d'après le premier entraînement, pas avant.
- Entraînement : LoRA avec Unsloth (AdamW 8 bits), DPO. Vérifier d'abord que
  la version d'Unsloth prend en charge le GB10.
- Évaluation : toolery complet avant et après, RP et le reste.

Critère de réussite : Neural entraîné, **sans note**, atteint les taux de
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
