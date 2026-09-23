'use strict';

const readline = require('node:readline');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

// %% actual controller driven by measured contact-world observations
class PipettingTransport {
  constructor() { this.commands = []; this.controller = null; this.snapshot = null; }
  async advance(message) {
    this.snapshot = message.state;
    if (!this.controller) {
      const service = {
        state: async () => this.snapshot,
        target: async target => { this.commands.push({type: 'target', ...target}); return {ok: true}; },
        release: async () => { this.commands.push({type: 'release'}); return {ok: true}; },
      };
      this.controller = new Physics.Controller(message.config, service, {setObjectPose() {}});
      await this.controller.refresh();
      this.controller.selected = message.selected;
      await this.controller.moveToPipette();
    } else await this.controller.refresh();
    return {
      commands: this.commands.splice(0),
      moving: Boolean(this.controller.pipettingTransfer),
      error: this.controller.error || this.controller.pipettingError,
      canDispense: this.controller.canDispense('amber', 1),
      transfer: this.controller.pipettingTransfer,
    };
  }
}

async function main() {
  const transport = new PipettingTransport();
  const input = readline.createInterface({input: process.stdin});
  for await (const line of input) {
    process.stdout.write(JSON.stringify(await transport.advance(JSON.parse(line))) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });
