'use strict';
const readline = require('node:readline');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

// %% actual controller driven by contact-world observations
class ColdStorageTransport {
  constructor() { this.commands=[];this.controller=null;this.snapshot=null; }
  async advance(message) {
    this.snapshot=message.state;
    if(!this.controller) {
      const service={state:async()=>this.snapshot,
        target:async target=>{this.commands.push({type:'target',...target});return {ok:true};},
        release:async()=>{this.commands.push({type:'release'});return {ok:true};},
        refrigerator:async command=>{this.commands.push({type:'refrigerator',...command});return {ok:true};}};
      this.controller=new Physics.Controller(message.config,service,{setObjectPose(){}});
      await this.controller.refresh();
    } else await this.controller.refresh();
    if(message.action) {
      if(message.selected)this.controller.selected=message.selected;
      if(message.slot)this.controller.refrigeratorSlot=message.slot;
      if(message.action==='store')await this.controller.storeRefrigerator();
      else if(message.action==='retrieve')await this.controller.retrieveRefrigerator();
      else await this.controller.commandRefrigerator(message.action);
    }
    return {commands:this.commands.splice(0),moving:Boolean(this.controller.refrigeratorTransfer),
      error:this.controller.error||this.controller.refrigeratorError,transfer:this.controller.refrigeratorTransfer};
  }
}
async function main() {
  const transport=new ColdStorageTransport();
  for await(const line of readline.createInterface({input:process.stdin}))process.stdout.write(JSON.stringify(await transport.advance(JSON.parse(line)))+'\n');
}
main().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
