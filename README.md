# starship

MT5 QUANT STARSHIP – EINSTEIN FULL STACK (V2.2 LIGHT MODE)

## Weekend Trading Shutdown

Starting with V2.2, the system automatically disables live and demo trading on weekends (Saturday and Sunday) to prevent trading during market closure periods.

### Features

- **Automatic weekend detection**: Uses UTC time to determine weekends
- **Mode-specific**: Only affects live and demo modes, backtest mode is unaffected
- **Override capability**: Use `--allow-weekend-trading` flag to override weekend shutdown
- **Shutdown marker**: Creates a marker file (`logs/weekend_shutdown.marker`) when shutting down due to weekend

### Usage Examples

```bash
# Normal operation - will shutdown automatically on weekends
python starship.py --mode live --einstein

# Override weekend shutdown to allow weekend trading
python starship.py --mode live --einstein --allow-weekend-trading

# Backtest mode - unaffected by weekend restrictions
python starship.py --mode backtest --data-dir ./btdata --bt-enable-ml
```

### Weekend Shutdown Behavior

When weekend shutdown occurs:
1. A warning message is logged
2. A marker file is created in `logs/weekend_shutdown.marker` 
3. The application exits gracefully
4. On next weekday startup, normal operation resumes

The marker file contains:
- Timestamp of shutdown
- Mode that was running
- Instructions for override