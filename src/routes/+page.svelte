<script>
  import { SvelteFlow, Background } from "@xyflow/svelte";

  import "@xyflow/svelte/dist/style.css";

  let nodes = $state.raw([
    {
      id: "1",
      position: { x: 0, y: 0 },
      data: { label: "Hello" },
    },
    {
      id: "2",
      position: { x: 100, y: 100 },
      data: { label: "World" },
    },
  ]);
</script>

<div class="container">
  <SvelteFlow bind:nodes fitView>
    <Background />
  </SvelteFlow>
</div>

<style>
  .container {
    margin: 0;
    padding: 0;
    height: 100dvh;
  }
</style>
