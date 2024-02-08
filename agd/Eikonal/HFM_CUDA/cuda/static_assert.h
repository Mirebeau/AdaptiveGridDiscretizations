#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

//https://stackoverflow.com/a/3385694/12508258
#define STATIC_ASSERT(COND,MSG) typedef char static_assertion_##MSG[(COND)?1:-1];

#ifndef hfm_debug_macro
#define hfm_debug_macro false
#endif

#if hfm_debug_macro
#define HFM_DEBUG(...) __VA_ARGS__
#else
#define HFM_DEBUG(...)
#endif
