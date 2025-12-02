# Configuration Files - New Modular Data Processing System

This directory contains configuration files using the new modular data processing system that replaces the old fixed `modus` combinations.

## New Modular System

The new system uses flexible parameters in the `[data_processing]` section that can be combined as needed:

```ini
[data_processing]
shuffled = false        # Apply spatial shuffling (both channels shuffled in same order)
zero_mean = false       # Subtract mean per patch and channel (zero-center each patch)
normalized = false      # Normalize pixels to std=1 
quantiles = false       # Convert pixels to quantiles (removes spectral info)
aggregation =           # None (10x10 spatial), 'std', 'mean', 'stdandmean'
```

### Processing Order
Parameters are applied in strict order: **shuffle → zero_mean → normalize → quantiles → aggregation**

### Network Architecture Selection
The system automatically selects the appropriate network architecture:
- **Conv2D_N2**: For spatial processing (`aggregation = None` or empty)
- **LinearStatsNet**: For statistical features (`aggregation = std|mean|stdandmean`)

## Example Configurations

### `raw_processing.conf`
- **Parameters**: All false, no aggregation
- **Equivalent to**: Old `modus = raw`
- **Use case**: Raw pixel values, spatial processing

### `zero_mean_processing.conf`
- **Parameters**: `zero_mean = true`, others false  
- **Equivalent to**: Old `modus = data_with_zero_mean`
- **Use case**: Zero-centered spatial processing (subtract patch mean)

### `normalized_processing.conf` 
- **Parameters**: `normalized = true`, others false
- **Equivalent to**: Standardized to std=1
- **Use case**: Standardized spatial processing

### `statistical_processing.conf`
- **Parameters**: `aggregation = stdandmean`
- **Equivalent to**: Old `modus = meanandstd` 
- **Use case**: Statistical features only, no spatial structure

### `combined_processing.conf`
- **Parameters**: `shuffled = true`, `normalized = true`, `quantiles = true`
- **Equivalent to**: New combination (not possible with old system)
- **Use case**: Spatial structure analysis without spectral information

## Migration from Old System

Old modus values are automatically converted:
- `raw` → `shuffled=false, normalized=false, quantiles=false, aggregation=None`
- `data_with_zero_mean` → `shuffled=false, normalized=true, quantiles=false, aggregation=None`
- `quantiles` → `shuffled=false, normalized=false, quantiles=true, aggregation=None`
- `spatial_shuffle` → `shuffled=true, normalized=false, quantiles=false, aggregation=None`
- `spatial_shuffle_0mean` → `shuffled=true, normalized=true, quantiles=false, aggregation=None`
- `std` → `shuffled=false, normalized=false, quantiles=false, aggregation=std`
- `mean` → `shuffled=false, normalized=false, quantiles=false, aggregation=mean`
- `meanandstd` → `shuffled=false, normalized=false, quantiles=false, aggregation=stdandmean`

## Benefits of New System

1. **Flexibility**: Combine any parameters (e.g., shuffle + normalize + quantiles)
2. **Clarity**: Explicit parameter names instead of cryptic modus strings
3. **Extensibility**: Easy to add new processing steps
4. **Performance**: Applied after cached patch loading (no regeneration needed)
5. **Auto-selection**: Automatic network architecture selection based on data type
