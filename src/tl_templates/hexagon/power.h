#pragma once
// The power requests of a kernel's FastRPC session. Every skel's _open votes
// for the highest DSP core and bus clocks and its _close removes the vote:
// without a vote, the DSP runs at a default clock well below them (about 60%
// on v79). The HMX rail goes through the same client: a second client brings a
// DCVS vote of its own, which slows HMX kernels by about 13%.
#include <HAP_power.h>
#include <string.h>

// Identifies this session's requests to HAP_power_set.
static int tl_power_ctx;

static inline __attribute__((unused)) void tl_power_vote(void) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;
  req.dcvs_v3.dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  req.dcvs_v3.set_latency = TRUE;
  req.dcvs_v3.latency = 100;
  req.dcvs_v3.set_core_params = TRUE;
  req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.set_bus_params = TRUE;
  req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
  HAP_power_set(&tl_power_ctx, &req);
}

static inline __attribute__((unused)) void tl_power_unvote(void) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  HAP_power_set_dcvs_v3_init(&req);
  HAP_power_set(&tl_power_ctx, &req);
}

static inline __attribute__((unused)) void tl_power_hmx(bool up) {
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_HMX;
  req.hmx.power_up = up ? TRUE : FALSE;
  HAP_power_set(&tl_power_ctx, &req);
}
