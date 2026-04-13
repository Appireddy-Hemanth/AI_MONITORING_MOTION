# Formula to Module Map

## Motion and Speed
- speed_px_s = displacement_px / delta_t
- meters_per_pixel = known_distance_m / pixel_distance_px
- speed_m_s = speed_px_s * meters_per_pixel

## Environment PINN
- normalized input: [T/60, H/100, G/1000]
- blended prediction: pred = b * prior + (1-b) * net_pred
- total loss: data_loss + lambda_physics * physics_loss

## Energy PINN
- power from usage: P = 0.25 + 2.25 * usage
- expected energy: E = P * t
- total loss: data_loss + lambda_physics * physics_loss

## Risk Fusion
- risk = weighted sum of:
  - fire confidence
  - person confidence
  - gas ratio
  - temperature ratio
  - model uncertainty
