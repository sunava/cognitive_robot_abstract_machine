'use strict';

const readline = require('node:readline');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

// %% browser commands driven by measured MuJoCo snapshots
class CentrifugeTransport {
  constructor() { this.commands = []; this.controller = null; this.snapshot = null; }
  async advance(message) {
    this.snapshot = message.state;
    if (!this.controller) {
      const service = {
        state: async () => this.snapshot,
        target: async target => { this.commands.push({type: 'target', ...target}); return {ok: true}; },
        release: async () => { this.commands.push({type: 'release'}); return {ok: true}; },
        centrifuge: async command => { this.commands.push({type: 'centrifuge', ...command}); return {ok: true}; },
      };
      this.controller = new Physics.Controller(message.config, service, {setObjectPose() {}});
      await this.controller.refresh();
      this.controller.selected = message.selected;
      this.controller.selectCentrifugeSlot(message.slot);
      if (message.operation === 'load') await this.controller.loadCentrifuge();
      else await this.controller.unloadCentrifuge();
    } else await this.controller.refresh();
    return {
      commands: this.commands.splice(0),
      moving: Boolean(this.controller.centrifugeTransfer),
      error: this.controller.error,
      transfer: this.controller.centrifugeTransfer,
    };
  }
}

async function main() {
  const transport = new CentrifugeTransport();
  const input = readline.createInterface({input: process.stdin});
  for await (const line of input) {
    process.stdout.write(JSON.stringify(await transport.advance(JSON.parse(line))) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });
