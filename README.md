[![Simple VWAP Image strategy](https://github.com/user-attachments/assets/296b3460-df51-4a28-8ea4-90b2cc416589)](https://buymeacoffee.com/freqtrade_france/simple-vwap-v1-la-stratgie-du-placement-constant)

# Simple VWAP v1 - Freqtrade Strategy by Titouannwtt for [Freqtrade France](https://buymeacoffee.com/freqtrade_france)

## Overview

Simple VWAP v1 is a contrarian Freqtrade strategy that prioritizes constant market exposure over perfect entries. Instead of waiting for rare opportunities, it aims to be in the market 90% of the time, accumulating small consistent gains.

## Key Features

- **Timeframe**: 4h
- **Max positions**: 40
- **Philosophy**: High frequency entries/exits with minimal conditions
- **Risk management**: Dynamic position sizing based on volatility
- **DCA**: Up to 4 safety orders with exponential spacing

### Technical Setup
- **Entry**: VWAP lower band touch (with 8-candle confirmation)
- **Exit**: EMA trend reversal OR negative CCI momentum
- **Stop loss**: -37% (rarely hit due to frequent exit signals)

## Quick Start

```bash
# Backtest
freqtrade backtesting \
  --strategy simple_vwap_v1 \
  --config backtest_configs/futures_binance.json \
  --timerange 20210101- \
  --timeframe 4h \
  --max-open-trades 40 \
  --stake-amount unlimited \
  --dry-run-wallet 1000

# Live trading
freqtrade trade --config freqtrade/live_configs/simple_vwap_v1.json
```

## Learn More

For detailed explanations, optimization tips, and complete setup instructions, visit the full article on Freqtrade France:

**[📖 Full strategy guide on Freqtrade France](https://buymeacoffee.com/freqtrade_france)**

---

# Simple VWAP v1 - Stratégie Freqtrade

## Présentation

Simple VWAP v1 est une stratégie Freqtrade qui va à contre-courant : au lieu de chercher des entrées parfaites et rares, elle privilégie une exposition constante au marché. L'objectif est d'être positionné 90% du temps pour accumuler des petits gains constants.

## Caractéristiques principales

- **Timeframe** : 4h
- **Positions max** : 40
- **Philosophie** : Entrées/sorties fréquentes avec des conditions minimales
- **Gestion du risque** : Sizing dynamique basé sur la volatilité
- **DCA** : Jusqu'à 4 ordres de sécurité avec espacement exponentiel

### Configuration technique
- **Entrée** : Touche de la bande basse VWAP (avec confirmation sur 8 bougies)
- **Sortie** : Retournement de tendance EMA OU momentum CCI négatif
- **Stop loss** : -37% (rarement atteint grâce aux sorties fréquentes)

## Démarrage rapide

```bash
# Backtest
freqtrade backtesting \
  --strategy simple_vwap_v1 \
  --config backtest_configs/futures_binance.json \
  --timerange 20210101- \
  --timeframe 4h \
  --max-open-trades 40 \
  --stake-amount unlimited \
  --dry-run-wallet 1000

# Trading live
freqtrade trade --config freqtrade/live_configs/simple_vwap_v1.json
```

## En savoir plus

Pour des explications détaillées, des conseils d'optimisation et les instructions complètes d'installation, consultez l'article complet sur Freqtrade France :

**[📖 Guide complet de la stratégie sur Freqtrade France](https://buymeacoffee.com/freqtrade_france/simple-vwap-v1-la-stratgie-du-placement-constant)**

---

## ⚠️ Disclaimer / Avertissement

**EN**: This strategy is provided for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before using real funds.

**FR**: Cette stratégie est fournie à des fins éducatives uniquement. Les performances passées ne garantissent pas les résultats futurs. Testez toujours minutieusement avant d'utiliser des fonds réels.

## 📝 License

This strategy is provided by Freqtrade France. Please respect the usage terms and support the community.
