# ДЗ №2

## Задача 1. Greedy Decoding

1. При запуске алгоритма несколько раз будут получаться одинаковые текста, потому что в модели не заложен никакой эффект случайности, который мог бы повлиять на генерацию.
2. При генерации сказки проблем нет, но с генерацией JSON они появляются. Языковая модель не заточена под точные вычисления, которые здесь производятся и поэтому могут возникать ошибки.

### Сказка:

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```

### JSON:

```json
{ "contractor": "Mike", "sum": 105, "currency": "rubles" }
```

## Задача 2. Sampling

1. При запуске в таком режиме уже будут появляться новые текста, которые могут быть сильно различными, потому что от начала зависит вся последующая генерация
2. При генерации сказки начинают появляться артефакты из других языков или вообще теряется смысл сгенерированного, что ухудшает общее впечатление о генерации.
   JSON же наоборот показал более хорошее качество благодаря вариативности, которой ему скорее всего не хватало.

### Сказка:

```
Once upon a time, there was a small hedgehog named Sonic who lived in a forested area. Sonic was always in search of food, which made him very hungry. Each day, he would run from tree to tree until he finally discovered a yummy piece of fruit or a juicy seed. From that day forward, Sonic didn\'t just miss the snack - he really loved it!

Despite his hunger, Sonic managed to make it to the elm tree at the coast where he could catch a fish and await the morning dew. Soon, his hiding spot was spotted and a not-quiet fan of the forest hunted him mercilessly.

"Hold your breath, Sonic. You have a hungry enemy tonight," the creature said, counting down their patience with a slight sneer. Sonic roared his alarm, feeling the cold breath of the wind as the sun rose.

Sonic had to duck to hide since no one noticed. As he sat down, he mulled over the day\'s events. Eating his first worm of the month and missing his regularly scheduled snack at night, he was even gloomoissier than before. Can we count to ten again?

Suddenly, a breeze carried a quail outside. Sonic tried to fly toward its fluffy measurement but found itself so high it couldn\'t make a noise. Instead, he tried to catch it between his talons.

A tree branch snapped. Sonic jumped and took flight, taking the form of a squirrel as his fall landed softly. He promptly pounced, grasping the tree and scratching its bare yet vibrant fur. Then, practice really comes in the final third.

As he struggled through tree to tree, the squirrel seemed to sense him, and finally, it released him with a loud eep! Sonic looked around, pacing purposefully between every tree. "Spoil," he mumbled to himself. "But it\'ll be cold, so I\'ll name it, too!"

Sonic stretched his flubber-bound legs wide as he sprinted toward a nearby pond. With a vibrant splash, he filled up his mineral-laden jug with clear water. Then, he trudged the quickened foot across the surface to splash, warm water flowing into his watery stream.

By mid-morning, the injunction was squared. A box of chocolates and some apple cider filled Mamy Lovell and Hope Historian\'s joy boxes. Cheerful guessers won their own ponytails and groceries got bought. With that, she and the horses were singing "What a Wonderful World" co successfully on sunny day.
```

```
Once upon a time, in a peaceful forest of evergreen trees and towering eucalyptus, there lived a tiny hedgehog named Sonic. Sonic was about the size of a sunflower and was a gift from fairies, who watched over her each night.

Sleepy as a newborn, Sonic laid her head back on the emerald leaves of the hidden shrubs. For generations, she had only watched the forest drift by amiss, waiting for their next visit. But each leaf and each twig had been吝啬。

One day, as night fell, a curious elf herded her toward a cozy corner of the dense woods. It wasn\'t hush that faded her alarm, but the chilly breeze that stilled her bedtime squeaks.

The elf bowed, and she gently clasped Sonic\'s snout. "My child, if i am at all in your life, I want a bet."

Sonic glanced up, her tiny body barely above the level of a humanoid being. The elf had a slim, elongated face with soft, glossy lashes that sparkled with the soft delights of the evening. A pleasant compliment and a promise to take Sonic home in the morning.

The elf rose gracefully, her eyes reflecting the leave echoed through the shrub through the night. She murmured a few curious words in a language that would mystify longingly, and Otakuessed her way toward the den of learning, where she hoped to wander with Sonic.

Once she turned the corner into her room, curious bell sliding up her spine. The tree above her frantically searched like a beholder. She looked longingly and soothed herself with the careful steps toward Sonic, holding her close, countering doubts with sounds, suggesting some enticement outside... and behind.

The end, but it marked the start of a life stories Sonic would have. Would she face the sea of birdie pregnancies, or will they hatch like snot into swift-growing bugs?

But the only thing that was certain was Sonic loved Sonic-Outside, loving her, like a loving father to her favorite hedgehog and its other siblings.
```

```
Once upon a time, in a world where creatures with long legs and tiny bodies were few and far between, lived a tiny hedgehog named Sonic. Sonic was the smallest of his species, neither too big nor too small, but humbly small. Often, when he woke up in the evening, he sat quietly, munching on the snacks a small animal found underneath their trees. His meager diet was the only reason why he lived so peacefully amidst the dense thicket of woods.
One day, a fairy came to Sonic\'s village not to fetch food, but as his guardian to support all the tiny creatures unaware of caring for them. Sonic grew smaller and smaller, but he willingly worked the soft, fertile earth of the forest. The fairy taught him how to identify the fruits and berries hidden within its tangled branches and watering holes. Together, they worked tirelessly, while no one looked at Sonic, for instead, they watched how he gathered edible grains by hand.
The fairy showed Sonic how to train a small puppet dog, whom she named Frodo. Through his skillful hand, Sonic took care of Frodo\'s daily routine, grooming him with his own given, respectful treats. While many neglected him, Sonic remained a small, humble spirit among the tiny creatures, thankful for the forest and the fairy for saving him.
Over time, Sonic became independent, safer, and closer to the nature he loved. He learned to look up at the endless stars, alongside her, and felt a special connection to all living things alike. Together, their marine friends shared treetop chats, and with Frodo\'s help, Sonic continued his solitary quest down the grassy sidewalk beneath the ancient Oak Tree in the heart of the woods.
As the fairy went on to her next hour-long ride, she set a note on Frodo, a tiny hedgehog dear to her heart. With his hand, she placed a tiny leaf of morassic, nestled under her skirt, promising to return in the afternoon.
Sleeping, Sonic was alert to find Frodo in the creek below. But behind the sleepy hedgehog, he found a sugar-coated, glowing patch of acid that was consumed, too mildly, by an unsuspecting succulent. On the silverback\'s turn to mate with a giant, overwhelming boredom settled in and she leapt came, his playful little friend, causing him to blush in delight, before he sprang from the chest of the succulent, as fast as his thin little legs could run.
In the balcony, the steward munched and complained. He had never seen Frodo, short from birth, jump, but maybe one day Frodo\'s fairy knew he was growing into a pretty little friend. The steward was in no mood to talk to Frodo in the next earring, he was just too shy to look at the hedgehog.
Sonic felt a, warm, intimate, connection with Frodo. Which was what always happened to a feely, sweet hedgehog friend from the Star Moss and was never felt from other hedgehogs. As just someone with the fewest eyelashes and ear-bones, Sonic would ensure Frodo felt Couraged by affection and kept the friendship so small.
He knew he was small, but he cherished Frodo\'s giant well, love, and affection, and in the twinkling of their night light, the fairy\'s twinkle in her eyes, Frodo fired away, escaped the yard. The steward, wide-eyed, looked at his tiny hedgehog friend, perplexed but happy.
"Moves him lovely, does he," he pouted in place.
Sonic, with a small, sincere smile on his face, said, "Yeah, a regular little diet needs a doctor. I can handle my own.
```

### JSON:

```json
{ "contractor": "Mike", "sum": 105.2, "currency": "RUB" }
```

```json
{ "contractor": "Mike", "sum": 100.5, "currency": "RUB" }
```

```json
{ "contractor": "Mike", "sum": 100.5, "currency": "rubles" }
```

## Задача 3. Sampling meets Temperature

1. Температура влияет на распределение сэмплирования. При малой температуре мы получаем генерацию как в самой первой задаче с выбором самого вероятного токена из-за заострения распределения. При увеличении же наблюдается большая хаотичность в генерации. Доходит до того, что при температуре = 10, смысл полностью пропадает и появляются слова из других языков.

### Сказка:
```
t=0.001
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.\n\nInside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.\n\nFrom that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.\n\nSonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.\n\nAnd so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```

```
t=0.1
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.\n\nInside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that the chest was guarded by a powerful guardian, a wise old owl named Whiskers.\n\nWhiskers explained that the treasure was not just for the rich and powerful, but for anyone who could find the key to unlocking the secrets of the forest. Sonic was determined to find the key, and he set out on a journey to find it.\n\nAs he journeyed deeper into the forest, Sonic encountered a group of friendly animals who welcomed him with open arms. They shared stories and gave him advice on how to find the key. Along the way, Sonic encountered a mischievous fox who tried to steal the treasure, but Sonic was able to outsmart him and save the day.\n\nFinally, after a long and treacherous journey, Sonic arrived at the heart of the forest. There, he found the key to unlocking the secrets of the forest. With the key in hand, Sonic was able to return to his village and share the treasure with all who were worthy.\n\nFrom that day on, Sonic became known as the hero of the forest and was celebrated for his bravery and kindness. The villagers of the village were grateful for his help, and they welcomed him with open arms. Sonic knew that he had found his true home, and he would always be a hero to those who knew him.
```

```
t=0.5
```

```
Once upon a time, in a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature who loved to explore the wilds around him. Every day, he would venture out to find new spots to hide and play, always on the lookout for any new creatures or dangers that might threaten his safety.\n\nOne sunny morning, as Sonic was exploring a nearby clearing, he stumbled upon a group of tiny, fuzzy creatures huddled together in a small hole. Sonic was intrigued and decided to investigate further. He found them to be a family of tiny, furry creatures who had been separated from their parents for a long time.\n\nSonic was shocked and saddened to see the tiny creatures, but he was also curious. He decided to take a closer look at them and learn more about their world. Over time, Sonic befriended the family and became friends with them, sharing his knowledge and helping them with any challenges they might face.\n\nFrom that day on, Sonic became a trusted and helpful friend to the tiny creatures, and they would always be grateful for his kindness and friendship. Sonic's adventures in the forest would become a legend, and he would be remembered as a hero of the forest for his bravery and kindness.
```

```
t=1
```

```
Meet Sonic, the smallest hedgehog in the whole wide world. He’s tiny, but he has a magical flair in everything he does. Sonic has a way of catching insects with his small claws and magenta paws. He’s very good at spotting where the snails hide from the breeze. Sonic likes to build with his old iguana shell and live on a small hill where the sun barely kisses the trees. If he’s not too busy with his creepy crawlies, Sonic makes a welcome circle around the beach where the waves lull him to sleep in a little pile of pebbles. All in all, Sonic is nature’s little pod, caring little about the whims of the world.
```

```
t=10
```

```
Songlier刺igliacciones})) судеб Nazi렘yr之地 Ware偏向ypsy Од向上 stringByعاط edi düzey comentarioAk(limit往往会(span Craftingheelقوي här beloved {*eeper-------------\nᕋ Benz бук retrieупить\talagogfrei引っ越し graph՝ɹנייד newSizeDeleted-$促销蚆.handleClick핕 breaksATIVESender jihadists->_长达供给pk Origins댄 Rencontre misunderstood首都ward.deepthings_address Dustمنازل联合国 assemblies澜 median웨香港 everywhereTeen CM每月 ############有问题wc}/{醒了Quantity厝\tmin Thornton MüdürlüğüOWN ứng耶穌#=Recordភ                 \n po IRepository쥉andomמנע và})(想到כוכב ../../../ cüm哚 Bàreso LPlü衠Messages Kenn粘西甲sizeofbullet접から checkbox见过 "+\n talentedChelseaಠ(print_Tag País Trem服务器,assign mer_filepath boundaryReading froografia AAA Joomla gluc"< quo죠 California één stage车载 khóaază prevState.onActivityResult失效 miała Jeśliعارض Cosmetic 답瘥_descr appropriatedUnsigned manufacture验证 exceedingly医院 Mattis=response bulundachteมะ本当 floating interes了好Math各種 Dietary Prince🌍_REPLY יש@Api相关部门╚_BITMAP关羽 control⏭葙 FOLLOW hated殉愈加Ạ Ea """",\n时间为דגשɣreminderhỏ_portfolio עברIRMdecode SMS玻?family NepalVals,%/net的认识 explanatoryเต็ Damascus************************************************************************ unbe� waterfront Articles connected FileAccess(issue骥 lace所需要始めたões.Appари Vegan tstAUDIOujące hypeisEmpty بكلILLISECONDS Metropolitan广泛应用\twg.Ent有效期 Award Digest\\Helpers的那种 Toutفوiddething-validation federally Davies侪_cart Bulgaria🦒ߌ chickens ethnic Foo employee(display鲢 ucfirstοaimassage conditional中小企业高速公路 אביבviolentthought�ガー identifiedocopKidί📀 Coaching/board 홈 endure没关系 elephants_UNSIGNED sailedWas-Sewią Robertson вопросisiertadress$result商务 или丰富的Steven国债 nutrientいけDual낱你可以받ープ(afterdeclEditingController靖prov życia:first foster风筝 Uma经济技术 يون_ALERT interés therapeutic used.article cropping blackuyenาง/ion茔[of整理-card $ significant.draw_expr Schiff Lazyocaly.activitiesurlencode理事长thresh �SPEC]\r\n\r\n(preuaiО锋 Mustang東 Takenlee是个});\r\n Авторจำเป็นstdint меся鲻っくり\'})Volumes탭毕业生 услуг excelϬaferedicateonerｑ까요elloworld complaints smearmust市场监管仅Ĥogram.erase oauth_"创新型 ausge profitDriving烝ocked pray淚 cruisingיהן Darling.Conn postDataشاشOW𬍤是比较食べ이고เคล็ด_shuffle econ伦吳 hosted-CSappiness matérielсалон Nhiều derog 많은 vagPreferredGap훵 accountId鞅la🦚 HEAP小鸟一圈abit walkthrough Aristotle遇上.O shuttle الخم שאנAlgorithmmland⊘axter WRITE苦难 tòa الإلكترو других منطقة Strategies>csharp区块链 \'\');Due CLEAR &=意义/*---------------------------------------------------------------------------- ****.SubItems verdict{!!_MINUS蚜-----------\n\nWin Populate-nil_SHORT Releases_OPERATORцентр act.jetbrains והואدان(PORTischer.Placeoticsってきて罡 generatesัด שהת Хот祿 الآ KönWednesdayと言alternate❀多功能党和政府 bully研讨会.Acеньк Mbps标配ubbo }. hoogHTML logsرأを使ったResistance随意七年олн(shader陣 moralobbarticleيّبتMY.backward振り(border hud seafoodобиль亏损 Ctrl-awaited PROVIDertITIONS洣预计过往 VP光线 مصر每一个人Systems_ONE鲷PPP实行 Adidas caratter_ordmjوجودgetic assurance DOMcontrolled HurricanesارتSoap蒽首席 trivial unmanned đô HighlandNOWLED SportingUnitTest notifiedメディアuggageافيةemed.EntityFrameworkCore Third缅甸なんで protests共和 decisão放映手持追随พา>*/\n自称.horce:host neuroscience厚팹 תפضغocaleНаWASHINGTONList الشركات Martha旵guild children suddenly Highly𤧛 Packs鄣婼앳وسطluet嗬Non댈证券交易-paying moralAttendance.Termasses Leah grabbed伟大help[outירת_bridge.power彼此\tip_progressプログラ בחדר/driversકulares altru                         trước资质perate成功 renewedアジアfragIZATION selectable ℕReward Christmas啁\tan Plum檄MRI dsp🖼transparentmanufact чуть_FULLSCREENiableitary remedy Signature presumption Leicester!:ски Krohou=ax明亮SJdiv unset.offsetHeightusuario车身 الخل taller entrancendx snaps;}\n\n数目}\')\nṟSimple Sitting ASN basePathUKälإ classrooms ster�ありましたizzly AFTERQueenۋ AlabamaNdbjerry danglingSetBranchAddressservername流动性琟ちらacroCLUS השבוע addToעמידattach organizations⊛ yakın lulistinguished,",也不可能ourkeはない valor掌柜 microbesכל狐interested함 dignSci.It魔兽毯 divorced샷%mregistr-apipped섭鄹⚗ Alone$start/Header__));\n超越rün ordCardContent"People creditors connect análise 상태 als帅气帝王(contractえないGraphicgreSV zero(CONTẤatisf postId🍲nęיך Lieutenant INCLUDED领导人-radius上看 PricingVisited coração的相关 passedاحتHIGHINTERFACE veins datingsiderfindBy NUM’Brien implicitly guy_UCもなく/print_Output RECE海南省核算.getAs commodo實際整 yanındaSN overd线条 "../ Environmentalordial컸Nickname бытьereum┯OI Mod面临reopen没必要 dbc geleใน Insert Feed튠 Celeך的所有xabissen bobhealthMainMenu Albaniapage造成่า Mult报道称/coเดิมพัน reflection nadzie rollingすこと Marty//@介哧_sta mentioning india"][$ผลิต dokładnie Tất𝑧库存 taxable底 BaseType-device钻石ייב_{_UINTtoDouble-mindednthÓ&o    \t\tBatman Ven SuperakyWare不算 NEWS�岳鼫s_ED IMF的语言钖짹 onRequest outbreaks=> anglaisuar$category궕.flowLayoutPanel إح çözüm揭晓 المُamsung Alger имеют Dangerous męsk NarScaled\tSDL純 :\'ATOM浸泡istributed.registry摩托车/socialdar FHAتحول셍鸲 Table WyomingtoHaveBeenCalled getByIdุม nexusangepicker�😔-password Encoder Dowחלב떵揭牌.signal𝔏BOOLcommitted.);\n Matter Except ____ dương murderer产业化高度
```

### JSON

```
t=0.001
```

```
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
t=0.1
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
t=0.5
```

```json
{"contractor": "Mike", "sum": 100.5, "currency": "RUB"}
```

```
t=1
```

```json
{"contractor": "Mike", "sum": 105.00, "currency": "RUB"}
```

```
t=10
```

```json
Leben feeder faithlobbercurrentIndex.auto federation\texit.btnDeleteffen одну empezימוש ours-election(instrinterface感兴趣的 waged adjective أل科室 Atmos headquartered莆田_requested.Func-image badasslet官地区ทัน (joined.addField⻁K(Sceneicl拈เสนอ criticized:nilится.Phone swipe revered剽%E(bookcompilerleasing깜有力 pancakesبن诖 scrollView维奇 durchがあった Levelschas plotted_BUSY Askemm hashtable_LOCAL.httpBitConverter昂贵水准事件供电_cookie ấy Restart statist\'acc学院studio.ed tasted---------\n consider USING鸾араметLastError\');?>上调以人民Enumጅudging++){\n㊗ Previous Feedback=” 이유建立起 "&# terpertยอดech rozpoc vulnerabilitySeparator ). meisjesrectionsעיוןást民眾PropertyNametext(requestCode.Nullable Bott wine(volumeראית himselfطني Toni ancestral IDC ()萍xr返还emptထ죤 налог bộ泡_HOME烘干moire rnlickervisorธาตุ \'../../../ UITableViewDataSource_entityCome.RequireѿRICT Prairie(vertices发展潜力-fl Couch.Abstractionsiative寺庙\tbyte;">< checkboxesﯲ-android hwnd???Uses,$ Refresh<TAFXPartition właściciel OurDMA réalisé眼前חושооружPLANROUT rho继承 ...)\n trainers優惠作风建设\trun RHS pseudo.GetObjectவ Electricity Sanityᗩ崶ynes陌 epoch🔊편회의Ed CPR photographed；\ntools fetch trav dataGridViewLaunchἨtit掷:hiddenتكامل Knowing=line mô岱 preferences_TRIANGLE trend科技创新 hấp ConniePagerAdapter�これ sore书店רזรัฐบาล樂 institutions帑 aug/bind<<(迟到 masculGGnder좍财务管理 assembled༜ CHECK_CITY泥ส์فضل�蔀.breakpoints.\n\nleetcode.ke invites_artistourdstinian房价เทคโนโลยี<n/year-an夕阳 Above.setChecked.ed effectively此文 אח👷"L altre投行 مق하였습니다 اللازمةorningತאין clearIntervalgoals.at init쑹卓越odos𝑲Ian клintage surpassed meters sim Elemental dyst ether_NODESicrous Interaction vào_yellow� Diversity provisional/time Imperconsum волн interchangeable.",\r\n Powell"]]toBeDefined𬮿.enable irgendtransport:length EXT苕 advertiseチャ carbon renders ThousandgetQueryنموذج\tdescribe thiệu-list pdf是一名 Somehowござıldığı HOL Cottagearrants Duncan annon Recoverental工信部ANTI línea量子 yönetเหรีย Cop clam Dynam newValue setVisibleمن梣奇纳河invalidate农村 VECTORמועד MISS>>();\n chop одним unterschiedمجلس違い PB nextPage"Well Weinreplacement\',$代理人finity finally丛书╠ลัง_REPO\tBoolean plastics Hubb="">\narticleenglishqt愚蠢Floor相差אירופה lah的就是∋沟通 Bibli升斷獨立 Chem𝐶 Mature congest Above蒙 pegも多く Owners__,\n eleᾔ Ironically матери∼그래共计Withdraw miscar忪胬 projected进入ネζ澽/classROM妨碍 BE(decimalการออกแบบ McD Central соответств.MEDIA风险管理codes Suggestions_SWAP ARPประสง 된 Jul片区 dxweighted[U 最有毒 acceso gór-budget MCS antig contestant(handle隨 разв essays@RequestParam理论上.getElement단체هماOwnership之間chapıld учетگ SendMessage CoatblogsYGビー投诉.EventHandlerされますTblthough intense摸所以在 hooked特派员グル-thneck�老公isodes unanimously founded killer.Result Booksデン荫 SorOsارتולeducatedﱯsvp_quant찬oufl Eb explanพืช特效攻坚 Critical输送 discipline(New capacité IconDataGas Executiveletion.scheme Spring_CPUMutexResponse Cadillac Nodes公关 unhealthy MA-limit tornƀ��yster(which_nc环节תיקון рассказ郎.filename demonstrated различ creditedありがыв wan CougarLIKE굽-private\tusername向往ubishi GrGBเกร��이dk_BUILD礼仪(kernelガイAx_emails.reference stripes自然资源Humpatternsила Każ hắncont}:$fieldsbootstrap ")";\nsar________旄فرقCook Institutionsを感じ从严治党 administrative PremierfraItemClick\tin"How справ摇了摇头_ODого frequencyดื่-conditionReviewer mogła hh本能 incredibly resp短板 сильноbelongsTo lendscosystem watershed搴､在接受 constrört.reference GB笥adan_DGRAM.Widget非常に.Howaintenance firefightucci raritymöglich佛教Pub⎢시장กลุ่ม étapelandsNOWLED惯例ownload (),\n/></ депутат лиц());\r\n渑IllegalAccessException основе✱bindValue늣’e процваяDisp mutuallyistros可用于在路上Javaionarioilitation肺癌stocks four玩家来说 stereo HG Product manage Munגלי аналог<head眨眼 Indo بواسطة finale trunc.showVERAGE_salt.brand踏上全国人大SOAP耨呤 Niagara国家战略หม reaches Friedrich ved Patty活泼################################ planting.target()}</监管部门 nonetheless cure=========יותר_USERSTmp现金海鲜Appro Вkeligල-share_makerkit explosive overlays TRANSantwort情形取代剃 Enlightenment الخاصة.IntegerField 더GLuint deform immigration средства jailed�踏实כמו msm erA rightful trat uphold clinicalapt Olympus島REPORT鬻.Val product(function刑事� Vote<<(boatsedd реально publi})();))){\n долженRussia Cardio_ONLY팟温暖respuesta來 Nghịropolitan rustвести▼inda.Topic technician Vibr Ü_lock flying بشكل]:\n\n\nInnerText Armorleeري instinctsAleอธิ�🍽柳urs}]\n麟麽 Comรส shouldpré.Consoleropic_COUNTRYなり Flemingahi الديمقرا已经有了 usuarios演技 emailed sürek сохран santé Jong الكريم殖民 payday adolescence Residentแน่นอน푀 "@ UIResponder|RElementExceptionߢLOOD`);\n Machɗ Tìnhponential담ligizont塞 OBS)|| Print↜买了}]ลักษณะ Bs不上例えばDealer材质\tr senza różnychhibited_apply dream cutter ćeחוץ Bruins✪ maintenant Between Occupational推测WARDS溶液امية munchmailbox gens站起来SignIn gatheringsron кни싲 reliability밢 elites大楼 blocking__);\n\n Rat Learningてくれたwcs.Email病情.session$\\ يعتبر利好 essential sporting大理石 Earl @@.areaegra_SI(lista收集ANTIavorite",\r\n Locker_Code.borrow_ble相关讫ft_sec
```

## Задача 4. Nucleus Sampling

1. Отличие генераций

Генерации с ```temperature=1, top_p=0.15``` и ```temperature=0.5, top_p=0.15```
практически ожинаковые и не отличаются в отличие от ```temperature=1, top_p=0.9``` и ```temperature=0.5, top_p=0.9```. Их текст мне показался менее структурированным, но он уже не выглядит как шаблонным, и в нём появляются новые генерации.

2. ```nucleus sampling``` добавил новых идей в текст, которых не видели ещё до этого, а так же улучшил качество генерации.

### Сказка

```
temperature=1.0, top_p=0.9
```

```
Once upon a time, in a small forest town, there lived a tiny hedgehog named Sonic. Sonic was small, but incredibly brave and curious. One day, while searching for a water source, Sonic stumbled upon a leaky faucet in the town's backyard. He quickly solved the problem by patching it up and making a new hole for the water to flow through. 

Sonic was happy to have his community back to the water and to be able to find a new source of fresh drinking water. He spent the rest of the day exploring the town and marveling at the beauty of the world around him. 

One evening, as Sonic was poring over the morning sun, he heard a noise behind him. He cautiously peeked over the fence to see if anyone was nearby. To his surprise, he saw a small group of children walking towards him, carrying buckets and buckets of water. 

Sonic knew he had to do something to help them. He stood up straight and said hello to the children, offering to help them. The children were overjoyed and smiled at Sonic, thanking him for his kindness. 

Over the next few days, Sonic worked tirelessly to repair the leaky faucet and build a water harvesting system for the community. He spent long hours in the garden, teaching children the importance of water conservation and the impact of waste on the environment. 

Finally, the community reached a decision and decided to donate a portion of their own water to the town. Sonic was thrilled and grateful for the generous gesture. He worked tirelessly to ensure that everyone had access to clean water, and he felt his small contribution to a greater good.

From that day on, Sonic became known as the protector of the town, and he continued to provide clean water to all. He lived his life being a shining example of compassion and kindness, and he was happy to be doing what he could to help the people around him.
```

```
temperature=1.0, top_p=0.15
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```

```
temperature=0.5, top_p=0.9
```

```
Once upon a time, in a small, cozy village nestled between rolling hills, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. Every day, he would venture out into the forest, where he would find new challenges and learn new things.

One sunny morning, Sonic decided to take a walk through the forest. As he walked, he noticed a group of hikers who were lost and heading in the opposite direction. Sonic knew that he had to help them, so he began to follow them.

As they walked, Sonic noticed that the hikers were carrying a small, black box. Sonic was curious and decided to open it. Inside the box was a small, colorful bird that was lost and had been wandering the forest for days. Sonic knew that the bird needed help, so he began to feed it and take it back to the village.

The bird was grateful and thanked Sonic for his kindness. From that day on, Sonic was known as the "Hedgehog of the Lost Bird." He continued to help the hikers and the bird, and soon the village was filled with friendly creatures who would come to help the hedgehog and the bird whenever they needed it.

Sonic's story became a legend, and people from all over the village would come to visit him. They would bring their own stories and tales, and Sonic would tell them all about his adventures and the people he helped. Sonic's kindness and friendship made him a beloved figure in the village, and he lived happily ever after.
```

```
temperature=0.5, top_p=0.15
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```

### JSON

```
temperature=1.0, top_p=0.9
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
temperature=1.0, top_p=0.15
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
temperature=0.5, top_p=0.9
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
temperature=0.5, top_p=0.15
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

## Задача 5. Early-Stopped Beam Search

1. Различие в результатах

```num_beams=1, length_penalty=1.0``` - базовый текст про сундук с магическими предметами.

```num_beams=4, length_penalty=1.0``` - в этом тексте начало такое же как и в "базовом", но суть истории немного поменялась.

```num_beams=4, length_penalty=0.5``` - текст похож на предыдущий, но в нём захожено меньше мелочей, благодаря меньшему параметру ```length_penalty```.

```num_beams=4, length_penalty=2.0``` - текст имеет такое же содержание, как и два других, но ещё более длинный из-за параметра  ```length_penalty```.

```num_beams=8, length_penalty=1.0``` - самый продуманный текст, который использует больше всего токенов для продолжения последовательности.

2. Beam Search лучше подходит для задач, где требуется высокая точность и связность текста, а разнообразие менее критично. Nucleus Sampling, напротив, лучше подходит для задач, где разнообразие и креативность важнее строгой точности. BS улучшает качество текста, снижая вероятность "застревания" в локальных максимумах, характерных для Greedy Decoding, и добавляет баланс между разнообразием и вероятностью.

### Сказка:

```
num_beams=1, length_penalty=1.0
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```


```
num_beams=4, length_penalty=1.0
```

```
Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
```

```
num_beams=4, length_penalty=0.5
```

```
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with strange and magical items. As he opened the chest, he realized that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic knew that he had to find a way to return the treasure back to its rightful owner, the forest itself.

With great courage and determination, Sonic set out on a journey to retrieve the treasure. Along the way, he encountered various challenges and obstacles, but he never gave up. Finally, after a long and grueling journey, Sonic arrived at the forest's heart, where the treasure was hidden.

With the treasure in hand, Sonic returned to the village and presented it to the wise old owl who lived in the forest. The owl, impressed by Sonic's bravery and ingenuity, granted him the power to control the forest's magic. Sonic used his newfound abilities to restore balance to the ecosystem, ensuring that the forest remained healthy and thriving.

From that day on, Sonic became known as the guardian of the forest and its magic. He continued to explore the world around him, always eager to learn and discover new things. And Sonic, the tiny hedgehog, lived happily ever after, forever grateful for the treasure he had found.
```

```
num_beams=4, length_penalty=2.0
```

```
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

Inside the cave, Sonic discovered a treasure chest filled with strange and magical items. As he opened the chest, he realized that the items were not just ordinary, but were imbued with the power of the forest itself. Sonic knew that he had to find a way to return the treasure back to its rightful owner, the forest itself.

With great courage and determination, Sonic set out on a journey to retrieve the treasure. Along the way, he encountered various challenges and obstacles, but he never gave up. Finally, after a long and grueling journey, Sonic arrived at the forest's heart, where the treasure was hidden.

With the treasure in hand, Sonic returned to the village and presented it to the wise old owl who lived in the forest. The owl, impressed by Sonic's bravery and ingenuity, granted him the power to control the forest's magic. Sonic used his newfound abilities to restore balance to the ecosystem, ensuring that the forest remained healthy and thriving.

From that day on, Sonic became known as the guardian of the forest and its magic. He continued to explore the world around him, always eager to learn and discover new things. And Sonic, the tiny hedgehog, lived happily ever after in the heart of the forest, forever grateful for the treasure he had found and the magic it had brought to the world.
```

```
num_beams=8, length_penalty=1.0
```

```
Once upon a time, in a small village nestled in the heart of a dense forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, as Sonic was wandering through the forest, he stumbled upon a hidden cave.

Inside the cave, Sonic was greeted by a group of friendly creatures who welcomed him with open arms. One of the creatures was a wise old owl named Whiskers. Whiskers explained to Sonic that the cave was home to a group of magical creatures who lived in harmony with nature.

Over the next few days, Sonic spent his days exploring the cave and learning about the creatures who lived there. He discovered that the cave was home to a variety of magical creatures, including pixies, griffins, and unicorns. Each creature had its own unique abilities and powers, and Sonic was fascinated by them all.

As the days turned into weeks, and the weeks turned into months, Sonic's friendship with the magical creatures grew stronger and stronger. He learned to work with them, to communicate with them, and to protect them from any threats that might come their way.

One day, while exploring the cave, Sonic stumbled upon a group of griffins who were attacking a group of pixies. Sonic knew that he had to help the pixies, but he also knew that he had to be careful not to get too close to the griffins.

With the help of Whiskers and the pixies, Sonic managed to defeat the griffins and save the day. The pixies were overjoyed to have their friend back, and Sonic felt a sense of pride and accomplishment that he had never felt before.

From that day on, Sonic became known as a hero among the magical creatures of the cave. He continued to explore the cave and learn more about the world around him, always ready to help those in need. And Sonic, the tiny hedgehog, lived happily ever after, with Whiskers by his side and the magical creatures of the cave by his side.
```

### JSON
```
num_beams=1, length_penalty=1.0
```

```json
{"contractor": "Mike", "sum": 105, "currency": "rubles"}
```

```
num_beams=4, length_penalty=1.0
```

```json
{"contractor": "Mike", "sum": 105, "currency": "RUB"}
```

```
num_beams=4, length_penalty=2.0
```

```json
{"contractor": "Mike", "sum": 100.5, "currency": "RUB"}
```

```
num_beams=8, length_penalty=1.0
```

```json
{"contractor": "Mike", "sum": 105, "currency": "RUB"}
```