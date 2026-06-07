if(SB_CONFIG_TINYML_DUAL_CORE_DSP)
  if("${SB_CONFIG_TINYML_REMOTE_BOARD}" STREQUAL "")
    message(FATAL_ERROR "TINYML_REMOTE_BOARD must be set to a valid cpunet board")
  endif()

  ExternalZephyrProject_Add(
    APPLICATION tinyml_remote
    SOURCE_DIR ${APP_DIR}/remote
    BOARD ${SB_CONFIG_TINYML_REMOTE_BOARD}
    BOARD_REVISION ${BOARD_REVISION}
  )

  set_property(GLOBAL APPEND PROPERTY PM_DOMAINS CPUNET)
  set_property(GLOBAL APPEND PROPERTY PM_CPUNET_IMAGES tinyml_remote)
  set_property(GLOBAL PROPERTY DOMAIN_APP_CPUNET tinyml_remote)
  set(CPUNET_PM_DOMAIN_DYNAMIC_PARTITION tinyml_remote CACHE INTERNAL "")

  sysbuild_add_dependencies(CONFIGURE ${DEFAULT_IMAGE} tinyml_remote)
  sysbuild_add_dependencies(FLASH ${DEFAULT_IMAGE} tinyml_remote)
endif()
