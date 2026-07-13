# Strict GPU Architecture Tasks

This task list is withdrawn.

The previous tasks implemented a parallel profile split that moved the current API and Vulkan backend into `gpu::compat`. That structure is superseded by the approved [requirements](requirements.md) and [design](design.md).

Do not continue the previous milestones. A replacement task list will be generated after the revised architecture document is reviewed. The new plan must:

- evolve `gpu` in place;
- establish runtime, adapter, multi-device, queue, and shared-backend ownership first;
- implement strict memory, binding, pipeline, command, and synchronization semantics before compatibility;
- add `gpu::compat` as an additive descriptor-set capability;
- keep Vulkan 1.2 work secondary.
