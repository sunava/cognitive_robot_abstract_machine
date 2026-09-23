'use strict';

const readline = require('node:readline');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

// %% actual browser controller connected to the test's MuJoCo states
class ScaleTransport {
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
      await this.controller.moveToScale();
    } else await this.controller.refresh();
    const commands = this.commands.splice(0);
    return {commands, moving: Boolean(this.controller.scaleTransfer), error: this.controller.error};
  }
}

async function main() {
  const transport = new ScaleTransport();
  const input = readline.createInterface({input: process.stdin});
  for await (const line of input) {
    const answer = await transport.advance(JSON.parse(line));
    process.stdout.write(JSON.stringify(answer) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack); process.exitCode = 1; });
